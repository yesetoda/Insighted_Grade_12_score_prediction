from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from datetime import datetime
import io
import logging
import gc
import os
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Performance Prediction API", 
              description="API for predicting student test scores using CatBoost models",
              version="1.0.0")

# All categorical features that were used during training
categorical_features = [
    'Gender', 'Health Issue', 'Career Interest', "Father's Education",
    "Mother's Education", 'Parental Involvement', 'Home Internet Access',
    'Electricity Access', 'School Type', 'School Location', 'Field Choice',
    'Has Textbook'
]

# Define the list of expected features for each track
common_subjects = [
    'Grade 12 - Civics and Ethical Education Test Score',
    'Grade 12 - Affan Oromoo Test Score',
    'Grade 12 - English Test Score',
    'Grade 12 - HPE Test Score',
    'Grade 12 - ICT Test Score'
]

natural_subjects = [
    'Grade 12 - Math for Natural Test Score',
    'Grade 12 - Biology Test Score',
    'Grade 12 - Chemistry Test Score',
    'Grade 12 - Physics Test Score',
    'Grade 12 - Agriculture Test Score',
    'Grade 12 - Technical Drawing Test Score'
]

social_subjects = [
    'Grade 12 - Math for Social Test Score',
    'Grade 12 - History Test Score',
    'Grade 12 - Geography Test Score',
    'Grade 12 - Economics Test Score',
    'Grade 12 - General Business Test Score'
]

# All categorical features that were used during training (base list)
base_categorical_features = [
    'Gender', 'Health Issue', 'Career Interest', "Father's Education",
    "Mother's Education", 'Parental Involvement', 'Home Internet Access',
    'Electricity Access', 'School Type', 'School Location', 'Field Choice'
]

# Binary features that need to be converted from Yes/No to 1/0
binary_features = [
    'Home Internet Access', 'Electricity Access', 'Has Textbook',
    'Health Issue'
]

# Model paths - using environment variables allows for easy configuration
NATURAL_MODEL_PATH = os.environ.get('NATURAL_MODEL_PATH', './catboost_natural_track.cbm')
SOCIAL_MODEL_PATH = os.environ.get('SOCIAL_MODEL_PATH', './catboost_social_track.cbm')

# Log the actual paths being used
logger.info(f"Natural model path: {NATURAL_MODEL_PATH}")
logger.info(f"Social model path: {SOCIAL_MODEL_PATH}")

# Check if model files exist
if not os.path.exists(NATURAL_MODEL_PATH):
    logger.warning(f"Natural model file not found at: {NATURAL_MODEL_PATH}")
if not os.path.exists(SOCIAL_MODEL_PATH):
    logger.warning(f"Social model file not found at: {SOCIAL_MODEL_PATH}")

# Memory cache for model info (doesn't store the models themselves)
model_info_cache = {}

# Define the request schema
class PredictionRequest(BaseModel):
    field_choice: str
    gender: str
    age: int
    health_issue: str
    career_interest: str 
    fathers_education: str
    mothers_education: str
    parental_involvement: str
    home_internet_access: str
    electricity_access: str
    school_type: str
    school_location: str
    has_textbook: str
    additional_features: Optional[Dict[str, Any]] = None

class MixedPredictionRequest(BaseModel):
    natural_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Data for Natural track students")
    social_data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Data for Social track students")

class PredictionResponse(BaseModel):
    field_choice: str
    predicted_scores: Dict[str, float]

@app.get("/")
async def root():
    return {"message": "Student Performance Prediction API"}

@app.get("/health")
async def health():
    """
    Health check endpoint that verifies models are accessible but doesn't load them
    """
    natural_exists = os.path.exists(NATURAL_MODEL_PATH)
    social_exists = os.path.exists(SOCIAL_MODEL_PATH)
    
    natural_info = f"File exists at {NATURAL_MODEL_PATH}" if natural_exists else f"File NOT found at {NATURAL_MODEL_PATH}"
    social_info = f"File exists at {SOCIAL_MODEL_PATH}" if social_exists else f"File NOT found at {SOCIAL_MODEL_PATH}"
    
    return {
        "status": "ok" if natural_exists and social_exists else "warning",
        "models": {
            "natural": {
                "exists": natural_exists,
                "path": NATURAL_MODEL_PATH,
                "info": natural_info
            },
            "social": {
                "exists": social_exists,
                "path": SOCIAL_MODEL_PATH,
                "info": social_info
            }
        }
    }

def load_model(track):
    """
    Lazy load a model only when needed
    """
    model = CatBoostRegressor()
    model_path = NATURAL_MODEL_PATH if track.lower() == "natural" else SOCIAL_MODEL_PATH
    
    try:
        # Log before loading
        logger.info(f"Attempting to load {track} model from: {model_path}")
        
        # Check if file exists and has content
        if not os.path.exists(model_path):
            logger.error(f"{track} model file not found at: {model_path}")
            return None
            
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            logger.error(f"{track} model file exists but is empty (0 bytes): {model_path}")
            return None
            
        logger.info(f"{track} model file exists and is {file_size/1024/1024:.2f} MB")
        
        # Try to load the model
        model.load_model(model_path)
        logger.info(f"{track} model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading {track} model: {e}")
        # Log more detailed diagnostic information
        logger.error(f"Model path: {model_path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Files in current directory: {os.listdir('.')}")
        return None

def convert_binary_features(df):
    for col in binary_features:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].map({
                'yes': 1, 'y': 1, 'true': 1, '1': 1, 'available': 1, 
                'no': 0, 'n': 0, 'false': 0, '0': 0, 'unavailable': 0,
                'none': 0, 'nan': 0
            })
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def preprocess_features_for_prediction(df):
    processed_df = df.copy()
    
    column_map = {
        'Has Textbook': 'Has Textbook',
        'HasTextbook': 'Has Textbook',
        'has_textbook': 'Has Textbook',
        "Father's Education": "Father's Education",
        'fathers_education': "Father's Education",
        "Mother's Education": "Mother's Education",
        'mothers_education': "Mother's Education",
        'field_choice': 'Field Choice'
    }
    
    processed_df.rename(columns={k: v for k, v in column_map.items() if k in processed_df.columns}, inplace=True)
    
    if 'Date of Birth' in processed_df.columns:
        try:
            current_year = datetime.now().year
            processed_df['Date of Birth'] = pd.to_datetime(processed_df['Date of Birth'], errors='coerce').dt.year
            processed_df['Age'] = current_year - processed_df['Date of Birth']
            processed_df.drop(columns=['Date of Birth'], inplace=True)
        except Exception as e:
            processed_df.drop(columns=['Date of Birth'], inplace=True, errors='ignore')
    
    processed_df = convert_binary_features(processed_df)
    
    for col in processed_df.columns:
        is_textbook_col = 'Textbook' in col
        if col in base_categorical_features or is_textbook_col:
            processed_df[col] = processed_df[col].astype(str)
            processed_df[col] = processed_df[col].fillna('Unknown')
            processed_df[col] = processed_df[col].replace('', 'Unknown')
            processed_df[col] = processed_df[col].str.strip()
        elif pd.api.types.is_numeric_dtype(processed_df[col]):
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                if processed_df[col].isnull().all():
                    processed_df[col] = 0
                else:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            except Exception as e:
                processed_df.drop(columns=[col], inplace=True, errors='ignore')
    
    exclude_cols = common_subjects + natural_subjects + social_subjects + ['Student ID', 'School ID', 'Total Test Score']
    feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
    
    return processed_df[feature_cols]

def get_predictions(df, field_choice):
    # Only load the model for the specific field choice
    model = load_model(field_choice)
    
    if field_choice == "Natural":
        track_subjects = natural_subjects
    else:
        track_subjects = social_subjects
    
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model for {field_choice} track could not be loaded")
    
    try:
        logger.info(f"DataFrame info before prediction:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Dynamic categorical features handling
        cat_features_in_df = [col for col in base_categorical_features if col in df.columns]
        textbook_cols = [col for col in df.columns if 'Textbook' in col]
        cat_features_in_df += textbook_cols
        
        for col in df.columns:
            if col not in cat_features_in_df:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().all():
                        df[col] = 0
                    else:
                        df[col] = df[col].fillna(df[col].median())
                except Exception as e:
                    df[col] = 0
        
        prediction_pool = Pool(
            data=df,
            cat_features=cat_features_in_df
        )
        
        predictions = model.predict(prediction_pool)
        
        all_subjects = common_subjects + track_subjects
        
        results = []
        if len(df) == 1:
            predictions_dict = {}
            if len(predictions.shape) == 2 and predictions.shape[1] == len(all_subjects):
                for i, subject in enumerate(all_subjects):
                    predictions_dict[subject] = float(predictions[0][i])
            else:
                predictions_dict["Total Score"] = float(predictions[0])
            results = predictions_dict
        else:
            for i in range(len(df)):
                pred_dict = {}
                if len(predictions.shape) == 2 and predictions.shape[1] == len(all_subjects):
                    for j, subject in enumerate(all_subjects):
                        pred_dict[subject] = float(predictions[i][j])
                else:
                    pred_dict["Total Score"] = float(predictions[i])
                results.append(pred_dict)
        
        # Free memory
        del model, prediction_pool, predictions
        gc.collect()
        
        return results
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Ensure model is explicitly deleted to free memory
        if 'model' in locals():
            del model
            gc.collect()

def process_batch(df, field_choice, batch_size=50):
    """Process dataframe in batches to conserve memory"""
    results = []
    num_batches = (len(df) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        logger.info(f"Processing batch {i+1}/{num_batches} with {len(batch_df)} records")
        
        processed_batch = preprocess_features_for_prediction(batch_df)
        batch_predictions = get_predictions(processed_batch, field_choice)
        
        # Free memory
        del processed_batch
        gc.collect()
        
        if isinstance(batch_predictions, dict):  # Single prediction case
            results.append(batch_predictions)
        else:  # Multiple predictions
            results.extend(batch_predictions)
            
    return results

@app.post("/predict_mixed_csv")
async def predict_mixed_csv(
    file: UploadFile = File(...)
):
    """
    Upload a single CSV file containing mixed data (both Natural and Social tracks).
    The API will automatically separate and process each track accordingly.
    """
    try:
        # Read the CSV file
        content = await file.read()
        
        try:
            # Try to read as CSV first
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception as csv_error:
            logger.warning(f"Failed to read as standard CSV: {str(csv_error)}")
            # Try with different encodings or delimiters
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')), encoding='latin1')
            except Exception:
                # Try with tab delimiter
                try:
                    df = pd.read_csv(io.StringIO(content.decode('utf-8')), delimiter='\t')
                except Exception as tab_error:
                    logger.error(f"All CSV reading attempts failed: {str(tab_error)}")
                    raise HTTPException(status_code=400, detail="Unable to parse the uploaded file. Please ensure it's a valid CSV file.")
        
        # Check if the dataframe is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
        
        # Ensure Field Choice column exists
        if 'Field Choice' not in df.columns:
            # Check for variations of the column name
            field_choice_variations = ['field_choice', 'Field_Choice', 'fieldchoice', 'field choice']
            found = False
            for variant in field_choice_variations:
                if variant in df.columns:
                    df.rename(columns={variant: 'Field Choice'}, inplace=True)
                    found = True
                    break
            
            if not found:
                # Try to add the field based on other indicators
                if 'Track' in df.columns:
                    df['Field Choice'] = df['Track'].apply(
                        lambda x: 'Natural' if str(x).lower() in ['natural', 'science', 'n', 'nat'] else 'Social'
                    )
                elif any('Natural' in col for col in df.columns) and not any('Social' in col for col in df.columns):
                    df['Field Choice'] = 'Natural'
                elif any('Social' in col for col in df.columns) and not any('Natural' in col for col in df.columns):
                    df['Field Choice'] = 'Social'
                else:
                    raise HTTPException(status_code=400, 
                                       detail="CSV must include 'Field Choice' column with values 'Natural' or 'Social'")
        
        # Standardize the Field Choice values
        df['Field Choice'] = df['Field Choice'].astype(str).apply(
            lambda x: 'Natural' if x.lower() in ['natural', 'science', 'n', 'nat'] else 'Social'
        )
        
        results = []
        
        # Define batch size based on available memory (smaller batches = less memory usage)
        BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '50'))
        
        # Process Natural track data
        natural_df = df[df['Field Choice'] == 'Natural']
        if len(natural_df) > 0:
            logger.info(f"Processing {len(natural_df)} Natural track students in batches")
            
            natural_predictions = process_batch(natural_df, "Natural", BATCH_SIZE)
            
            for i, pred in enumerate(natural_predictions):
                results.append({
                    'Index': i,
                    'Field Choice': 'Natural',
                    'Student ID': str(natural_df['Student ID'].iloc[i]) if 'Student ID' in natural_df.columns else f"Natural_{i}",
                    'Predicted Scores': pred
                })
            
            # Clear memory
            del natural_df, natural_predictions
            gc.collect()
        
        # Process Social track data
        social_df = df[df['Field Choice'] == 'Social']
        if len(social_df) > 0:
            logger.info(f"Processing {len(social_df)} Social track students in batches")
            
            social_predictions = process_batch(social_df, "Social", BATCH_SIZE)
            
            # Calculate offset for indices
            offset = len(natural_df) if 'natural_df' in locals() else 0
            
            for i, pred in enumerate(social_predictions):
                results.append({
                    'Index': offset + i,
                    'Field Choice': 'Social',
                    'Student ID': str(social_df['Student ID'].iloc[i]) if 'Student ID' in social_df.columns else f"Social_{i}",
                    'Predicted Scores': pred
                })
            
            # Clear memory
            del social_df, social_predictions
            gc.collect()
        
        if not results:
            raise HTTPException(status_code=400, 
                               detail="No valid records found with Field Choice as 'Natural' or 'Social'")
        
        # Free up memory before returning
        del df, content
        gc.collect()
        
        return JSONResponse(content={"predictions": results})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in predict_mixed_csv: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Error processing mixed CSV file: {str(e)}")

@lru_cache(maxsize=1)
def get_model_info(track):
    """
    Retrieve model info with caching to avoid reloading models
    """
    if track in model_info_cache:
        return model_info_cache[track]
    
    model = load_model(track)
    if model is None:
        return {"error": f"Failed to load {track} model"}
    
    try:
        info = {
            "feature_names": model.feature_names_ if hasattr(model, 'feature_names_') else None,
            "tree_count": model.tree_count_ if hasattr(model, 'tree_count_') else None,
            "params": model.get_params()
        }
        
        # Cache the info
        model_info_cache[track] = info
        
        # Free memory
        del model
        gc.collect()
        
        return info
    except Exception as e:
        # Free memory
        del model
        gc.collect()
        return {"error": str(e)}

@app.get("/model-info")
async def model_info():
    """
    Returns information about the models without keeping them in memory.
    """
    natural_info = get_model_info("Natural")
    social_info = get_model_info("Social")
    
    return {
        "natural_model": natural_info,
        "social_model": social_info
    }

@app.get("/required-features")
async def required_features():
    """
    Returns information about required features for predictions
    """
    return {
        "common_features": [
            "Gender", "Age", "Health Issue", "Career Interest", 
            "Father's Education", "Mother's Education", "Parental Involvement",
            "Home Internet Access", "Electricity Access", "School Type", 
            "School Location", "Has Textbook", "Field Choice"
        ],
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "common_subjects": common_subjects,
        "natural_subjects": natural_subjects,
        "social_subjects": social_subjects
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
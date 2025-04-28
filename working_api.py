
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from datetime import datetime
import io
import json
import logging

# # All categorical features that were used during training
categorical_features = [
    'Gender', 'Health Issue', 'Career Interest', "Father's Education",
    "Mother's Education", 'Parental Involvement', 'Home Internet Access',
    'Electricity Access', 'School Type', 'School Location', 'Field Choice',
    'Has Textbook'
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Performance Prediction API", 
              description="API for predicting student test scores using CatBoost models",
              version="1.0.0")

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

# Load the trained models
logger.info("Loading models...")
natural_model = CatBoostRegressor()
social_model = CatBoostRegressor()

try:
    natural_model.load_model('catboost_natural_track.cbm')
    social_model.load_model('catboost_social_track.cbm')
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Error loading models: {e}")

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
    return {
        "status": "ok", 
        "models": {
            "natural": natural_model is not None, 
            "social": social_model is not None
        }
    }

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
    if field_choice == "Natural":
        model = natural_model
        track_subjects = natural_subjects
    else:
        model = social_model
        track_subjects = social_subjects
    
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model for {field_choice} track is not loaded")
    
    try:
        logger.info(f"DataFrame info before prediction:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Data types: {df.dtypes}")
        
        # Dynamic categorical features handling
        cat_features_in_df = [col for col in base_categorical_features if col in df.columns]
        textbook_cols = [col for col in df.columns if 'Textbook' in col]
        cat_features_in_df += textbook_cols
        logger.info(f"Categorical features in DataFrame: {cat_features_in_df}")
        
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
        
        if len(df) == 1:
            predictions_dict = {}
            if len(predictions.shape) == 2 and predictions.shape[1] == len(all_subjects):
                for i, subject in enumerate(all_subjects):
                    predictions_dict[subject] = float(predictions[0][i])
            else:
                predictions_dict["Total Score"] = float(predictions[0])
            return predictions_dict
        else:
            predictions_list = []
            for i in range(len(df)):
                pred_dict = {}
                if len(predictions.shape) == 2 and predictions.shape[1] == len(all_subjects):
                    for j, subject in enumerate(all_subjects):
                        pred_dict[subject] = float(predictions[i][j])
                else:
                    pred_dict["Total Score"] = float(predictions[i])
                predictions_list.append(pred_dict)
            return predictions_list
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"DataFrame info for debugging:")
        logger.error(f"Shape: {df.shape}")
        logger.error(f"Columns: {df.columns.tolist()}")
        logger.error(f"Data types: {df.dtypes}")
        if len(df) > 0:
            sample_row = df.iloc[0].to_dict()
            logger.error(f"First row sample: {sample_row}")
            for col, val in sample_row.items():
                if not isinstance(val, (int, float, str, bool, type(None))):
                    logger.error(f"Column '{col}' has non-standard type: {type(val)}")
        raise HTTPException(status_code=500, detail=error_msg)

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
        
        # Print debug info about the CSV
        logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
        
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
        
        # Process Natural track data
        natural_df = df[df['Field Choice'] == 'Natural']
        if len(natural_df) > 0:
            logger.info(f"Processing {len(natural_df)} Natural track students")
            processed_natural_df = preprocess_features_for_prediction(natural_df)
            natural_predictions = get_predictions(processed_natural_df, "Natural")
            
            for i, pred in enumerate(natural_predictions):
                results.append({
                    'Index': i,
                    'Field Choice': 'Natural',
                    'Student ID': str(natural_df['Student ID'].iloc[i]) if 'Student ID' in natural_df.columns else f"Natural_{i}",
                    'Predicted Scores': pred
                })
        
        # Process Social track data
        social_df = df[df['Field Choice'] == 'Social']
        if len(social_df) > 0:
            logger.info(f"Processing {len(social_df)} Social track students")
            processed_social_df = preprocess_features_for_prediction(social_df)
            social_predictions = get_predictions(processed_social_df, "Social")
            
            for i, pred in enumerate(social_predictions):
                results.append({
                    'Index': len(natural_df) + i,
                    'Field Choice': 'Social',
                    'Student ID': str(social_df['Student ID'].iloc[i]) if 'Student ID' in social_df.columns else f"Social_{i}",
                    'Predicted Scores': pred
                })
        
        if not results:
            raise HTTPException(status_code=400, 
                               detail="No valid records found with Field Choice as 'Natural' or 'Social'")
        
        return JSONResponse(content={"predictions": results})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in predict_mixed_csv: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Error processing mixed CSV file: {str(e)}")

@app.get("/model-info")
async def model_info():
    """
    Returns information about the loaded models.
    """
    natural_info = {}
    social_info = {}
    
    if natural_model is not None:
        try:
            natural_info = {
                "feature_names": natural_model.feature_names_ if hasattr(natural_model, 'feature_names_') else None,
                "tree_count": natural_model.tree_count_ if hasattr(natural_model, 'tree_count_') else None,
                "params": natural_model.get_params()
            }
        except Exception as e:
            natural_info = {"error": str(e)}
    
    if social_model is not None:
        try:
            social_info = {
                "feature_names": social_model.feature_names_ if hasattr(social_model, 'feature_names_') else None,
                "tree_count": social_model.tree_count_ if hasattr(social_model, 'tree_count_') else None,
                "params": social_model.get_params()
            }
        except Exception as e:
            social_info = {"error": str(e)}
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

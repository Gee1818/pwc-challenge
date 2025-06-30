import streamlit as st
import polars as pl
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from feature_engineering import perform_feature_engineering_preds

# Set page config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model weights for ensemble (adjust these based on your model performance)
MODEL_WEIGHTS = {
    'rf': 0.3942,      # Random Forest: 39.42%
    'lasso': 0.3358,   # Lasso: 33.58% 
    'lgbm': 0.27       # LightGBM: 27.00%
}

@st.cache_resource
def load_models_and_pipelines():
    """Load all trained models and preprocessing pipelines."""
    models = {}
    pipelines = {}

    model_names = ['rf', 'lasso', 'lgbm']

    for model_name in model_names:
        # Try loading model
        model_path = f"../models/final_{model_name}_model.pkl"
        if os.path.exists(model_path):
            try:
                # Try standard pickle loading
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            except Exception as e1:
                try:
                    # Try with different pickle protocol
                    import pickle5 as pickle_alt
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle_alt.load(f)
                except ImportError:
                    try:
                        # Try with joblib (often used for sklearn models)
                        import joblib
                        models[model_name] = joblib.load(model_path)
                    except Exception as e2:
                        st.warning(f"Could not load {model_name} model: {str(e1)}. Also tried joblib: {str(e2)}")
                except Exception as e2:
                    st.warning(f"Could not load {model_name} model with pickle5: {str(e2)}")
        else:
            st.warning(f"Model file not found: {model_path}")
        
        # Try loading preprocessing pipeline
        pipeline_path = f"../models/{model_name}_pipeline.pkl"
        if os.path.exists(pipeline_path):
            try:
                # Try standard pickle loading
                with open(pipeline_path, 'rb') as f:
                    pipelines[model_name] = pickle.load(f)
            except Exception as e1:
                try:
                    # Try with different pickle protocol
                    import pickle5 as pickle_alt
                    with open(pipeline_path, 'rb') as f:
                        pipelines[model_name] = pickle_alt.load(f)
                except ImportError:
                    try:
                        # Try with joblib
                        import joblib
                        pipelines[model_name] = joblib.load(pipeline_path)
                    except Exception as e2:
                        st.warning(f"Could not load {model_name} pipeline: {str(e1)}. Also tried joblib: {str(e2)}")
                except Exception as e2:
                    st.warning(f"Could not load {model_name} pipeline with pickle5: {str(e2)}")
        else:
            st.warning(f"Pipeline file not found: {pipeline_path}")
    
    return models, pipelines

def create_input_dataframe(age, gender, education_level, job_title, years_experience):
    """Create a DataFrame from user inputs in the format expected by the models."""
    
    # Create a dictionary with the input features
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education_level],
        'Job Title': [job_title],
        'Years of Experience': [years_experience]
    }
    
    # Convert to DataFrame
    df = pl.DataFrame(input_data)
    df = perform_feature_engineering_preds(df)
    
    return df

def make_ensemble_prediction(models, pipelines, input_df, weights):
    """Make predictions using all available models and return weighted ensemble."""
    predictions = {}
    
    for model_name in models.keys():
        if model_name in pipelines:
            try:
                # Preprocess the input data
                X_processed = pipelines[model_name].transform(input_df)
                
                # Make prediction
                pred = models[model_name].predict(X_processed)[0]
                
                # For lgbm_log model, apply exponential transformation
                if model_name == 'lgbm_log':
                    pred = np.expm1(pred)  # Reverse log transformation
                
                predictions[model_name] = pred
                
            except Exception as e:
                st.warning(f"Error making prediction with {model_name}: {str(e)}")
    
    if not predictions:
        return None, predictions
    
    # Calculate weighted ensemble prediction
    ensemble_pred = sum(pred * weights.get(model_name, 0) for model_name, pred in predictions.items())
    
    return ensemble_pred, predictions

def main():
    st.title("💰 Salary Prediction App")
    st.markdown("---")
    
    # Add diagnostic section
    with st.expander("🔧 Diagnostic Information", expanded=False):
        st.write("**Python Version:**", sys.version)
        st.write("**Current Directory:**", os.getcwd())
        
        # Check if models directory exists
        models_dir = "models"
        if os.path.exists(models_dir):
            st.write("**Models Directory Contents:**")
            for file in os.listdir(models_dir):
                file_path = os.path.join(models_dir, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "N/A"
                st.write(f"• {file} ({file_size} bytes)")
        else:
            st.error(f"Models directory '{models_dir}' not found!")
    
    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write("**Ensemble Weights:**")
        for model, weight in MODEL_WEIGHTS.items():
            st.write(f"• {model.upper()}: {weight:.0%}")
        
        st.markdown("---")
        st.write("**Models Used:**")
        st.write("• Lasso Regression")
        st.write("• LightGBM")
        st.write("• LightGBM (Log Target)")
        st.write("• Random Forest")
        
        st.markdown("---")
        st.write("**Troubleshooting:**")
        st.write("If models won't load, try:")
        st.write("• Check if saved with joblib")
        st.write("• Verify Python versions match")
        st.write("• Check file permissions")
    
    # Load models
    with st.spinner("Loading models..."):
        models, pipelines = load_models_and_pipelines()
    
    if not models:
        st.error("❌ No models could be loaded. Please ensure model files are in the 'models/' directory.")
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(models)} models")
    
    # Create input form
    st.header("📝 Enter Your Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            help="Your current age"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Select your gender"
        )
        
        education_level = st.selectbox(
            "Education Level",
            options=[
                "Bachelor's",
                "Master's",
                "PhD",
                "None",

            ],
            index=1,
            help="Your highest level of education"
        )
    
    with col2:
        job_title = st.text_input(
            "Job Title",
            placeholder="Enter your job title (e.g., Software Engineer)",
            help="Enter your job title"
        )
        
        years_experience = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=5,
            help="Total years of professional experience"
        )
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Salary", type="primary", use_container_width=True):

        if education_level == 'None':
            education_level ==''

        # Create input DataFrame
        input_df = create_input_dataframe(age, gender, education_level, job_title, years_experience)
        
        # Make predictions
        with st.spinner("Making predictions..."):
            ensemble_pred, individual_preds = make_ensemble_prediction(
                models, pipelines, input_df, MODEL_WEIGHTS
            )
        
        if ensemble_pred is not None:
            # Display results
            st.header("🎯 Prediction Results")
            
            # Main prediction
            st.metric(
                label="💰 Predicted Annual Salary",
                value=f"${ensemble_pred:,.0f}",
                help="Ensemble prediction from all models"
            )
            
            # Individual model predictions
            st.subheader("📈 Individual Model Predictions")
            
            pred_cols = st.columns(len(individual_preds))
            
            for i, (model_name, pred) in enumerate(individual_preds.items()):
                with pred_cols[i]:
                    weight = MODEL_WEIGHTS.get(model_name, 0)
                    st.metric(
                        label=f"{model_name.upper()}",
                        value=f"${pred:,.0f}",
                        delta=f"Weight: {weight:.0%}"
                    )
       
            # Confidence note
            st.info("💡 **Note:** This prediction is based on trained machine learning models and should be used as a reference. Actual salaries may vary based on company, location, and other factors not captured in this model.")
            
        else:
            st.error("❌ Could not make predictions. Please check if all models are properly loaded.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

@st.cache_resource 
def train_churn_model(df_processed):
    """
    Trains the churn prediction model using a Logistic Regression classifier
    within a scikit-learn pipeline. Handles preprocessing and class imbalance.

    Args:
        df_processed (pd.DataFrame): The DataFrame after initial feature engineering.

    Returns:
        tuple: A tuple containing:
            - model_pipeline (sklearn.pipeline.Pipeline): The trained scikit-learn pipeline.
            - X_test (pd.DataFrame): Features of the test set (preprocessed).
            - y_test (pd.Series): Target labels of the test set.
            - numerical_features (list): List of original numerical feature names.
            - categorical_features (list): List of original categorical feature names.
    """
    st.info("Preparing data and training the Logistic Regression model...")

    if 'is_churned' not in df_processed.columns:
        st.error("Error: 'is_churned' column not found in the processed data. Cannot train model.")
        return None, None, None, None, None

    X = df_processed.drop('is_churned', axis=1)
    y = df_processed['is_churned']

    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

   
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' 
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.write(f"Original training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

   
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

  
    all_feature_names_preprocessed = preprocessor.get_feature_names_out()

    st.write("Applying SMOTE to handle class imbalance on preprocessed training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    st.write(f"Resampled training data shape: {X_train_resampled.shape}")
    st.write(f"Original training churn distribution: {y_train.value_counts()}")
    st.write(f"Resampled training churn distribution: {y_train_resampled.value_counts()}")

 
    model_pipeline = Pipeline(steps=[('classifier', LogisticRegression(random_state=42, solver='liblinear'))])

    
    model_pipeline.fit(X_train_resampled, y_train_resampled)
    st.success("Model training complete.")

    
    model_pipeline.preprocessor_ = preprocessor
    model_pipeline.feature_names_ = all_feature_names_preprocessed

    return model_pipeline, X_test_preprocessed, y_test, numerical_features, categorical_features

def get_feature_importance(model_pipeline, numerical_features, categorical_features):
    """
    Extracts and organizes feature importance (coefficients) from the trained Logistic Regression model.
    Assumes the model_pipeline has a 'preprocessor_' and 'feature_names_' attribute.
    """
    st.info("Extracting feature importance...")

    
    all_feature_names = model_pipeline.feature_names_

    coefficients = model_pipeline.named_steps['classifier'].coef_[0]

    feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefficients})
    feature_importance_df['Absolute_Coefficient'] = np.abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

    st.success("Feature importance extracted.")
    return feature_importance_df

def predict_churn_for_new_data(model_pipeline, new_data_df):
    """
    Predicts churn probability for new, unseen customer data.

    Args:
        model_pipeline (sklearn.pipeline.Pipeline): The trained scikit-learn pipeline (should contain preprocessor_).
        new_data_df (pd.DataFrame): DataFrame containing new customer data (must have same columns as training data).

    Returns:
        pd.DataFrame: DataFrame with customer_id and predicted churn probability.
    """
    st.info("Predicting churn for new data...")

    
    expected_raw_cols = [
        'customer_id', 'join_date', 'last_purchase_date', 'total_purchases', 'total_spent',
        'avg_time_on_site_min', 'num_support_interactions',
        'product_category_preference', 'newsletter_subscribed', 'promo_clicks_last_month'
    ]

    missing_cols = [col for col in expected_raw_cols if col not in new_data_df.columns]
    if missing_cols:
        st.error(f"Missing required columns for prediction: {', '.join(missing_cols)}. Please ensure your input data has all necessary columns.")
        return pd.DataFrame()

    customer_ids = new_data_df['customer_id'].copy()


    from data_utils import engineer_features, handle_missing_data 

    
    new_data_df_cleaned = handle_missing_data(new_data_df.copy(), strategy='drop_rows')

    if new_data_df_cleaned.empty:
        st.warning("No valid data rows remaining after missing data handling for prediction.")
        return pd.DataFrame()

    
    if 'is_churned' not in new_data_df_cleaned.columns:
        new_data_df_cleaned['is_churned'] = 0 

    engineered_new_data = engineer_features(new_data_df_cleaned.copy())

    if 'is_churned' in engineered_new_data.columns:
        engineered_new_data = engineered_new_data.drop(columns=['is_churned'])

   
    preprocessor = model_pipeline.preprocessor_

    transformed_new_data = preprocessor.transform(engineered_new_data)

    churn_probabilities = model_pipeline.named_steps['classifier'].predict_proba(transformed_new_data)[:, 1]
    predicted_churn_labels = (churn_probabilities > 0.5).astype(int) 

    results_df = pd.DataFrame({
        'customer_id': customer_ids.loc[new_data_df_cleaned.index],
        'churn_probability': churn_probabilities,
        'predicted_churn': predicted_churn_labels
    })
    st.success("Churn prediction complete.")
    return results_df

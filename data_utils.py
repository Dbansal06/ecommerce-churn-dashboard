import pandas as pd
import numpy as np
import datetime
import random
import streamlit as st 

def generate_synthetic_data(num_customers=10000, churn_rate=0.15):
    """
    Generates a synthetic e-commerce customer dataset with churn labels.
    """
    st.info(f"Generating synthetic e-commerce data for {num_customers} customers...")

    base_date = datetime.datetime(2024, 1, 1)

    data = {
        'customer_id': range(1, num_customers + 1),
        'join_date': [base_date - datetime.timedelta(days=random.randint(30, 730)) for _ in range(num_customers)],
        'last_purchase_date': [base_date - datetime.timedelta(days=random.randint(0, 365)) for _ in range(num_customers)],
        'total_purchases': [random.randint(1, 50) for _ in range(num_customers)],
        'total_spent': [round(random.uniform(10, 1000), 2) for _ in range(num_customers)],
        'avg_time_on_site_min': [round(random.uniform(1, 60), 2) for _ in range(num_customers)],
        'num_support_interactions': [random.randint(0, 10) for _ in range(num_customers)],
        'product_category_preference': [random.choice(['Electronics', 'Apparel', 'Home Goods', 'Books', 'Groceries', 'Beauty']) for _ in range(num_customers)],
        'newsletter_subscribed': [random.choice([True, False]) for _ in range(num_customers)],
        'promo_clicks_last_month': [random.randint(0, 20) for _ in range(num_customers)],
    }

    df = pd.DataFrame(data)
    df['is_churned'] = 0

    num_churned = int(num_customers * churn_rate)
    churn_indices = np.random.choice(df.index, num_churned, replace=False)

    df.loc[churn_indices, 'is_churned'] = 1
    df.loc[churn_indices, 'last_purchase_date'] = df.loc[churn_indices, 'join_date'] + pd.to_timedelta([random.randint(10, 180) for _ in range(num_churned)], unit='D')
    df.loc[churn_indices, 'total_purchases'] = [random.randint(1, 5) for _ in range(num_churned)]
    df.loc[churn_indices, 'total_spent'] = [round(random.uniform(10, 100), 2) for _ in range(num_churned)]
    df.loc[churn_indices, 'avg_time_on_site_min'] = [round(random.uniform(1, 10), 2) for _ in range(num_churned)]
    df.loc[churn_indices, 'num_support_interactions'] = [random.randint(0, 2) for _ in range(num_churned)]
    df.loc[churn_indices, 'promo_clicks_last_month'] = [random.randint(0, 3) for _ in range(num_churned)]

    df['last_purchase_date'] = df.apply(lambda row: max(row['last_purchase_date'], row['join_date']), axis=1)

    st.success(f"Synthetic data generated for {num_customers} customers. Actual churn rate: {df['is_churned'].mean():.2f}")
    return df

@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file):
    """Loads data from an uploaded CSV file."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully from uploaded file!")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}. Please ensure it's a valid CSV.")
            return None
    return None

@st.cache_data(show_spinner=False)
def load_data_from_url(url):
    """Loads data from a public URL (CSV format expected)."""
    if url:
        try:
            df = pd.read_csv(url)
            st.success("Data loaded successfully from URL!")
            return df
        except Exception as e:
            st.error(f"Error loading data from URL: {e}. Please ensure the URL is correct and points to a CSV file.")
            return None
    return None

def handle_missing_data(df, strategy='drop_rows'):
    """
    Handles missing data based on the selected strategy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): 'drop_rows', 'drop_columns', 'impute_mean', 'impute_median', 'impute_mode'.

    Returns:
        pd.DataFrame: DataFrame with missing data handled.
    """
    st.info(f"Handling missing data using strategy: '{strategy.replace('_', ' ').title()}'")
    initial_rows, initial_cols = df.shape

    if strategy == 'drop_rows':
        df_cleaned = df.dropna()
        st.warning(f"Dropped {initial_rows - df_cleaned.shape[0]} rows with missing values.")
    elif strategy == 'drop_columns':
        df_cleaned = df.dropna(axis=1)
        st.warning(f"Dropped {initial_cols - df_cleaned.shape[1]} columns with missing values.")
    elif strategy in ['impute_mean', 'impute_median', 'impute_mode']:
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    if strategy == 'impute_mean':
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    elif strategy == 'impute_median':
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                elif pd.api.types.is_string_dtype(df_cleaned[col]) or pd.api.types.is_object_dtype(df_cleaned[col]):
                    if strategy == 'impute_mode':
                        
                        mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else None
                        if mode_val is not None:
                            df_cleaned[col].fillna(mode_val, inplace=True)
        st.success("Missing numerical data imputed. Categorical missing values (if any) are not imputed by this function but will be handled by OneHotEncoder's `handle_unknown='ignore'` if they appear in test set.")
    else:
        st.warning("Invalid missing data strategy. No action taken.")
        df_cleaned = df.copy() 

    st.write(f"Data shape after handling missing values: {df_cleaned.shape}")
    return df_cleaned

@st.cache_data
def engineer_features(df):
    """
    Performs feature engineering on the raw e-commerce data.
    """
    st.info("Performing feature engineering...")
    df_copy = df.copy()

    for col in ['join_date', 'last_purchase_date']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            if df_copy[col].isnull().any():
                st.warning(f"Warning: Missing or invalid dates found in '{col}'. Rows with invalid dates will be dropped or imputed based on missing data strategy.")
                
                df_copy.dropna(subset=[col], inplace=True)


    
    if not df_copy['last_purchase_date'].empty:
        current_date = df_copy['last_purchase_date'].max() + datetime.timedelta(days=30)
        df_copy['recency_days'] = (current_date - df_copy['last_purchase_date']).dt.days
    else:
        df_copy['recency_days'] = 0 


    
    if not df_copy['join_date'].empty:
        df_copy['tenure_days'] = (current_date - df_copy['join_date']).dt.days
    else:
        df_copy['tenure_days'] = 0


   
    df_copy['avg_purchase_value'] = df_copy.apply(
        lambda row: row['total_spent'] / row['total_purchases'] if row['total_purchases'] > 0 else 0,
        axis=1
    )

    df_copy['engagement_score'] = df_copy['avg_time_on_site_min'] + \
                                  (df_copy['num_support_interactions'] * 5) + \
                                  (df_copy['promo_clicks_last_month'] * 2)

    cols_to_drop = ['customer_id', 'join_date', 'last_purchase_date']
    df_processed = df_copy.drop(columns=[col for col in cols_to_drop if col in df_copy.columns])

    st.success("Feature engineering complete.")
    return df_processed


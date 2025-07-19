import streamlit as st
import pandas as pd
import numpy as np


from data_utils import generate_synthetic_data, load_data_from_upload, load_data_from_url, handle_missing_data, engineer_features
from model_utils import train_churn_model
from app_sections import (
    show_overview_section,
    show_eda_section,
    show_model_performance_section,
    show_model_interpretation_section,
    show_retention_strategies_section,
    show_prediction_section
)


st.set_page_config(layout="wide", page_title="E-commerce Churn Prediction Dashboard")

st.title("🛍️ E-commerce Customer Churn Prediction Dashboard")


st.sidebar.header("Data Source & Settings")

data_source = st.sidebar.radio(
    "Choose your data source:",
    ("Synthetic Data", "Upload CSV File", "Load from Public URL")
)

df_raw = None
if data_source == "Synthetic Data":
    st.sidebar.subheader("Synthetic Data Parameters")
    num_customers_input = st.sidebar.slider("Number of Synthetic Customers", 1000, 50000, 10000, 1000)
    churn_rate_input = st.sidebar.slider("Synthetic Churn Rate", 0.05, 0.50, 0.15, 0.01)
    df_raw = generate_synthetic_data(num_customers=num_customers_input, churn_rate=churn_rate_input)
elif data_source == "Upload CSV File":
    st.sidebar.subheader("Upload Your CSV Data")
    uploaded_file = st.sidebar.file_uploader("Drag and drop your CSV file here", type="csv")
    df_raw = load_data_from_upload(uploaded_file)
elif data_source == "Load from Public URL":
    st.sidebar.subheader("Load Data from Public URL")
    url = st.sidebar.text_input("Enter Public CSV URL", "https://raw.githubusercontent.com/plotly/datasets/master/Churn.csv")
    st.sidebar.info("Note: The example URL is a generic churn dataset. For this app, ensure your CSV has columns like: `customer_id`, `join_date`, `last_purchase_date`, `total_purchases`, `total_spent`, `avg_time_on_site_min`, `num_support_interactions`, `product_category_preference`, `newsletter_subscribed`, `promo_clicks_last_month`, `is_churned`.")
    df_raw = load_data_from_url(url)

if df_raw is not None and not df_raw.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("Data Preprocessing Options")
    missing_data_strategy = st.sidebar.selectbox(
        "Handle Missing Data:",
        ('drop_rows', 'drop_columns', 'impute_mean', 'impute_median', 'impute_mode'),
        help="Choose how to deal with missing values. 'Impute mode' is for categorical/string data, others for numerical."
    )

    df_cleaned = handle_missing_data(df_raw.copy(), strategy=missing_data_strategy)

    if 'is_churned' not in df_cleaned.columns:
        st.error("Error: The 'is_churned' (target) column is missing from your data after preprocessing. Please ensure your dataset includes it.")
        st.stop() 

   
    df_cleaned['is_churned'] = df_cleaned['is_churned'].astype(int)

   
    df_processed = engineer_features(df_cleaned.copy())

    
    model_pipeline, X_test, y_test, numerical_features, categorical_features = train_churn_model(df_processed.copy())

   
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 EDA",
        "⚙️ Model Performance",
        "🧠 Model Interpretation",
        "💡 Strategies",
        "🔮 Predict Churn"
    ])

    with tab1:
        show_overview_section(df_raw, df_processed)

    with tab2:
        show_eda_section(df_processed, numerical_features, categorical_features)

    with tab3:
        show_model_performance_section(y_test, y_pred, y_pred_proba)

    with tab4:
        show_model_interpretation_section(model_pipeline, numerical_features, categorical_features)

    with tab5:
        show_retention_strategies_section()

    with tab6:
        show_prediction_section(model_pipeline)

else:
    st.info("Please select a data source from the sidebar to begin.")
    st.markdown("---")
    st.markdown("### Data Requirements:")
    st.write("""
    For this dashboard to work, your dataset (synthetic or uploaded) should ideally contain the following columns:
    - `customer_id` (unique identifier)
    - `join_date` (date customer joined, YYYY-MM-DD format)
    - `last_purchase_date` (date of last purchase, YYYY-MM-DD format)
    - `total_purchases` (number of purchases)
    - `total_spent` (total money spent)
    - `avg_time_on_site_min` (average minutes spent on site)
    - `num_support_interactions` (number of times customer interacted with support)
    - `product_category_preference` (e.g., 'Electronics', 'Apparel')
    - `newsletter_subscribed` (True/False or 1/0)
    - `promo_clicks_last_month` (number of promotional clicks)
    - `is_churned` (0 for not churned, 1 for churned - this is your target variable)
    """)
    st.markdown("---")
    st.markdown("### Example Public Dataset (for testing):")
    st.write("You can try loading a generic churn dataset from this URL (though column names might differ):")
    st.code("https://raw.githubusercontent.com/plotly/datasets/master/Churn.csv")
    st.write("Remember, for the app's features to fully align, your custom data's columns should match the expected format.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import datetime


from data_utils import engineer_features
from model_utils import get_feature_importance, predict_churn_for_new_data


sns.set_style("whitegrid")

def show_overview_section(df_raw, df_processed):
    """Displays the overview and key metrics section of the dashboard."""
    st.header("📊 Overview & Key Metrics")
    st.markdown("""
    Welcome to the E-commerce Customer Churn Prediction Dashboard!
    This tool helps you understand customer behavior and predict who might leave, so you can take action to keep them.
    """)

    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total Customers", value=f"{len(df_raw):,}")
    with col2:
        churn_count = df_raw['is_churned'].sum()
        st.metric(label="Churned Customers", value=f"{churn_count:,}")
    with col3:
        actual_churn_rate = df_raw['is_churned'].mean() * 100
        st.metric(label="Actual Churn Rate", value=f"{actual_churn_rate:.2f}%")

    st.markdown("---")

    st.subheader("Understanding Your Data: Raw vs. Processed")
    st.write("""
    Here's a glimpse of the data we're working with.
    * **Raw Data:** This is how the data originally looks, with dates and customer IDs.
    * **Processed Data:** This is the data after we've transformed it into features our model can understand (like 'Recency' and 'Tenure').
    """)

    view_data_type = st.radio("Select data view:", ("Raw Data (Original)", "Processed Data (Features)"), horizontal=True)

    if view_data_type == "Raw Data (Original)":
        st.dataframe(df_raw.head())
        st.subheader("Raw Data Description")
        st.dataframe(df_raw.describe(include='all'))
    else:
        st.dataframe(df_processed.head())
        st.subheader("Processed Data Description")
        st.dataframe(df_processed.describe())

    st.markdown("---")
    st.subheader("What is Churn?")
    st.write("""
    In e-commerce, **customer churn** refers to customers who stop doing business with you.
    Predicting churn helps businesses proactively identify at-risk customers and implement strategies to retain them,
    which is often more cost-effective than acquiring new customers.
    """)


def show_eda_section(df_processed, numerical_features, categorical_features):
    """Displays the Exploratory Data Analysis (EDA) section."""
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section helps us visualize patterns in the data and understand how different factors relate to customer churn.
    """)

    st.subheader("Churn Distribution")
    st.write("This chart shows the proportion of customers who churned (left) versus those who did not.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='is_churned', data=df_processed, palette='viridis', ax=ax)
    ax.set_title('Distribution of Churn (0: Not Churned, 1: Churned)')
    ax.set_xlabel('Churn Status')
    ax.set_ylabel('Number of Customers')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Churned', 'Churned'])
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Numerical Feature Distributions by Churn Status")
    st.write("""
    Explore how the values of numerical features (like spending or time on site) are distributed for churned vs. non-churned customers.
    You might notice differences that help explain why some customers leave.
    """)
    selected_num_feature = st.selectbox(
        "Select a numerical feature to visualize its distribution:",
        numerical_features,
        key="num_feature_select_eda"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_processed, x=selected_num_feature, hue='is_churned', kde=True, palette='coolwarm', common_norm=False, ax=ax)
    ax.set_title(f'Distribution of {selected_num_feature.replace("_", " ").title()} by Churn Status')
    ax.set_xlabel(selected_num_feature.replace("_", " ").title())
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Scatter Plot: Recency vs. Total Spent")
    st.write("""
    This plot helps visualize the relationship between how recently a customer purchased (`Recency`)
    and their total spending (`Total Spent`), colored by their churn status.
    Often, churned customers are in the top-left (high recency, low total spent) area.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='recency_days', y='total_spent', hue='is_churned', data=df_processed, palette='coolwarm', alpha=0.6, ax=ax)
    ax.set_title('Recency vs. Total Spent by Churn Status')
    ax.set_xlabel('Recency (Days Since Last Purchase)')
    ax.set_ylabel('Total Spent')
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Engagement Score by Churn Status")
    st.write("""
    The 'Engagement Score' combines factors like time on site, support interactions, and promo clicks.
    This box plot shows the range and median of engagement scores for both churned and non-churned groups.
    Lower engagement often correlates with higher churn.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='is_churned', y='engagement_score', data=df_processed, palette='viridis', ax=ax)
    ax.set_title('Engagement Score by Churn Status')
    ax.set_xlabel('Churn Status (0: Not Churned, 1: Churned)')
    ax.set_ylabel('Engagement Score')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Churned', 'Churned'])
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Categorical Feature Distribution by Churn Status")
    st.write("""
    See how different categories (like product preference or newsletter subscription) are distributed
    among churned and non-churned customers.
    """)
    selected_cat_feature = st.selectbox(
        "Select a categorical feature to visualize its distribution:",
        categorical_features,
        key="cat_feature_select_eda"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x=selected_cat_feature, hue='is_churned', data=df_processed, palette='magma', ax=ax)
    ax.set_title(f'{selected_cat_feature.replace("_", " ").title()} by Churn Status')
    ax.set_xlabel(selected_cat_feature.replace("_", " ").title())
    ax.set_ylabel('Number of Customers')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)


def show_model_performance_section(y_test, y_pred, y_pred_proba):
    """Displays the model performance metrics and charts."""
    st.header("⚙️ Model Performance")
    st.markdown("""
    This section evaluates how well our churn prediction model is performing.
    """)

    st.subheader("Classification Report")
    st.write("""
    The Classification Report provides key metrics like Precision, Recall, and F1-score for both
    'Not Churned' (0) and 'Churned' (1) classes.
    * **Precision:** Out of all customers predicted to churn, how many actually churned?
    * **Recall:** Out of all customers who actually churned, how many did our model correctly identify?
    * **F1-Score:** A balance between Precision and Recall.
    """)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    st.markdown("---")

    st.subheader("Confusion Matrix")
    st.write("""
    The Confusion Matrix helps visualize the types of correct and incorrect predictions:
    * **True Negative (Top-Left):** Correctly predicted 'Not Churned'.
    * **False Positive (Top-Right):** Predicted 'Churned', but actually 'Not Churned' (Type I error).
    * **False Negative (Bottom-Left):** Predicted 'Not Churned', but actually 'Churned' (Type II error - often costly!).
    * **True Positive (Bottom-Right):** Correctly predicted 'Churned'.
    """)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Receiver Operating Characteristic (ROC) Curve")
    st.write("""
    The ROC curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
    The **Area Under the Curve (AUC)** is a single number summarizing the curve:
    * **AUC close to 1:** Excellent model, good separation between classes.
    * **AUC close to 0.5:** Model is no better than random guessing.
    """)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)


def show_model_interpretation_section(model_pipeline, numerical_features, categorical_features):
    """Displays the model interpretation (feature importance) section."""
    st.header("🧠 Model Interpretation (Feature Importance)")
    st.markdown("""
    This section helps us understand *why* the model makes its predictions.
    By looking at **Feature Importance**, we can see which factors are most influential in predicting churn.
    """)

    feature_importance = get_feature_importance(model_pipeline, numerical_features, categorical_features)

    st.subheader("Top 15 Most Important Features")
    st.write("""
    The table below shows the features that the model considers most important for predicting churn.
    * **Coefficient:** Indicates the direction and strength of the feature's impact.
        * **Positive Coefficient:** As this feature's value increases, the likelihood of churn increases.
        * **Negative Coefficient:** As this feature's value increases, the likelihood of churn decreases.
    * **Absolute_Coefficient:** The strength of the impact, regardless of direction.
    """)
    st.dataframe(feature_importance.head(15))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(15), palette='coolwarm', ax=ax)
    ax.set_title('Feature Importance (Coefficients) for Churn Prediction')
    ax.set_xlabel('Coefficient Value (Positive: Increases Churn, Negative: Decreases Churn)')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Key Takeaways from Feature Importance:**
    * **High Recency (Positive Coefficient):** Customers who haven't made a purchase recently are more likely to churn.
    * **Low Total Spent/Purchases (Negative Coefficient):** Customers who spend less or buy less frequently are more prone to churn.
    * **Low Engagement Score (Positive Coefficient):** Less engaged customers (less time on site, fewer promo clicks) are at higher risk.
    * **Product Category Preference:** Some categories might have higher churn rates than others, indicating specific issues or preferences.
    """)


def show_retention_strategies_section():
    """Displays the retention strategies section."""
    st.header("💡 Retention Strategy Suggestions")
    st.markdown("""
    Based on the insights from our churn prediction model, here are actionable strategies to reduce customer churn.
    These are general suggestions; a real-world application would involve A/B testing these strategies to measure their actual impact.
    """)

    st.subheader("1. Re-engage Inactive Customers (High Recency)")
    st.markdown("""
    * **Personalized Campaigns:** Send targeted emails or push notifications with exclusive discounts or product recommendations based on their past purchases/browsing.
    * **Win-back Offers:** Offer special incentives (e.g., "We miss you!" discount) to customers who haven't purchased in a while.
    * **Feedback Surveys:** Reach out to understand why they became inactive and address their concerns.
    """)

    st.subheader("2. Nurture Low-Frequency Buyers (Low Total Purchases)")
    st.markdown("""
    * **Subscription Models:** Encourage recurring purchases through subscription services for frequently bought items.
    * **Product Bundles:** Suggest complementary products to increase average order value and encourage repeat visits.
    * **Loyalty Programs:** Reward customers for consistent purchases, even small ones.
    """)

    st.subheader("3. Value High-Spending Customers (High Total Spent)")
    st.markdown("""
    * **VIP Programs:** Offer exclusive benefits, early access to sales, or premium customer support to your most valuable customers.
    * **Personalized Outreach:** Dedicated account managers or personalized communications to ensure their satisfaction.
    * **Exclusive Content:** Provide access to special content or events.
    """)

    st.subheader("4. Boost Overall Customer Engagement (Low Engagement Score)")
    st.markdown("""
    * **Improve UX:** Enhance website navigation, search functionality, and product discovery to make browsing more enjoyable.
    * **Interactive Content:** Create engaging blog posts, videos, or quizzes related to their interests.
    * **Personalized Communication:** Tailor marketing messages based on browsing behavior, not just purchase history, to keep them interested.
    """)

    st.subheader("5. Optimize Support Experience (High Support Interactions for Churners)")
    st.markdown("""
    * **Root Cause Analysis:** Investigate common issues leading to support interactions for churned customers and fix underlying problems.
    * **Faster Resolution:** Ensure customer queries are resolved quickly and effectively to prevent frustration.
    * **Proactive Support:** Anticipate potential issues and provide solutions before customers even reach out.
    """)

    st.warning("Remember: These are data-driven suggestions. Always validate strategies with A/B testing in a real business environment.")


def show_prediction_section(model_pipeline):
    """Allows users to input new customer data and get churn predictions."""
    st.header("🔮 Predict Churn for New Customers")
    st.markdown("""
    You can input details for a new customer or a set of customers below to see their predicted churn probability.
    This helps identify at-risk customers *before* they churn.
    """)

   
    prediction_mode = st.radio("How would you like to input data?", ("Single Customer Input", "Upload CSV File"), horizontal=True)

    if prediction_mode == "Single Customer Input":
        st.subheader("Enter Customer Details:")
        st.write("Provide the following information for a new customer:")

        
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID (e.g., C12345)", "New_Customer_1")
           
            join_date_input = st.date_input("Join Date", datetime.date(2023, 1, 1))
            last_purchase_date_input = st.date_input("Last Purchase Date", datetime.date(2024, 6, 15))
            total_purchases = st.number_input("Total Purchases (count)", min_value=0, value=10)
            total_spent = st.number_input("Total Spent ($)", min_value=0.0, value=250.0)
        with col2:
            avg_time_on_site_min = st.number_input("Avg Time on Site (min)", min_value=0.0, value=15.0)
            num_support_interactions = st.number_input("Number of Support Interactions", min_value=0, value=1)
            product_category_preference = st.selectbox("Favorite Product Category", ['Electronics', 'Apparel', 'Home Goods', 'Books', 'Groceries', 'Beauty'])
            newsletter_subscribed = st.checkbox("Newsletter Subscribed?", True)
            promo_clicks_last_month = st.number_input("Promo Clicks Last Month", min_value=0, value=5)

   
        new_customer_data = pd.DataFrame([{
            'customer_id': customer_id,
            'join_date': join_date_input.strftime('%Y-%m-%d'),
            'last_purchase_date': last_purchase_date_input.strftime('%Y-%m-%d'),
            'total_purchases': total_purchases,
            'total_spent': total_spent,
            'avg_time_on_site_min': avg_time_on_site_min,
            'num_support_interactions': num_support_interactions,
            'product_category_preference': product_category_preference,
            'newsletter_subscribed': newsletter_subscribed,
            'promo_clicks_last_month': promo_clicks_last_month,
            'is_churned': 0 
        }])

        if st.button("Predict Churn for this Customer"):
            with st.spinner("Calculating prediction..."):
                prediction_results_df = predict_churn_for_new_data(model_pipeline, new_customer_data)
                if not prediction_results_df.empty:
                    st.subheader("Prediction Result:")
                    churn_prob = prediction_results_df['churn_probability'].iloc[0]
                    predicted_churn_label = "CHURN" if prediction_results_df['predicted_churn'].iloc[0] == 1 else "NO CHURN"

                    st.markdown(f"**Customer ID:** `{prediction_results_df['customer_id'].iloc[0]}`")
                    st.markdown(f"**Predicted Churn Probability:** `{churn_prob:.2%}`")
                    if predicted_churn_label == "CHURN":
                        st.error(f"**Predicted Status:** {predicted_churn_label} ⚠️")
                        st.write("This customer is predicted to churn. Consider immediate retention efforts!")
                    else:
                        st.success(f"**Predicted Status:** {predicted_churn_label} ✅")
                        st.write("This customer is predicted to stay. Keep up the good work!")

                    st.subheader("Raw Input Data for Prediction")
                    st.dataframe(new_customer_data)
                    st.subheader("Engineered Features for Prediction (for this input)")
                
                    engineered_display_df = engineer_features(new_customer_data.drop(columns=['is_churned']))
                    st.dataframe(engineered_display_df)

    elif prediction_mode == "Upload CSV File":
        st.subheader("Upload a CSV file with new customer data:")
        st.write("""
        The CSV file should have the following columns (case-sensitive):
        `customer_id`, `join_date` (YYYY-MM-DD), `last_purchase_date` (YYYY-MM-DD),
        `total_purchases`, `total_spent`, `avg_time_on_site_min`,
        `num_support_interactions`, `product_category_preference`,
        `newsletter_subscribed` (True/False), `promo_clicks_last_month`.
        """)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            new_customers_df = pd.read_csv(uploaded_file)
            st.write("Uploaded data preview:")
            st.dataframe(new_customers_df.head())

            if st.button("Predict Churn for Uploaded Data"):
                with st.spinner("Calculating predictions for uploaded data..."):
                    
                    if 'is_churned' not in new_customers_df.columns:
                        new_customers_df['is_churned'] = 0

                    prediction_results_df = predict_churn_for_new_data(model_pipeline, new_customers_df.copy()) 

                    if not prediction_results_df.empty:
                        st.subheader("Prediction Results for Uploaded Customers:")
                        st.dataframe(prediction_results_df)

                        st.download_button(
                            label="Download Predictions as CSV",
                            data=prediction_results_df.to_csv(index=False).encode('utf-8'),
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )

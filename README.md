🛍️ E-commerce Customer Churn Prediction Dashboard
This project provides an interactive Streamlit dashboard to understand, predict, and strategize against customer churn in an e-commerce context. It allows you to use synthetic data, upload your own CSV, or load data from a public URL. The dashboard includes sections for data overview, exploratory data analysis (EDA), model performance, model interpretation (Explainable AI), retention strategies, and even a live prediction tool for new customers.

✨ Features
Flexible Data Input: Use a built-in synthetic data generator, upload your own CSV file, or provide a public CSV URL.

Missing Data Handling: Options to drop rows/columns with missing values or impute them (mean, median, mode).

Automated Feature Engineering: Calculates key e-commerce metrics like Recency, Tenure, Average Purchase Value, and an Engagement Score.

Churn Prediction Model: Trains a Logistic Regression model with SMOTE for class imbalance handling.

Interactive Dashboard:

Overview: Key metrics and sample data.

EDA: Visualizations (histograms, scatter plots, box plots, count plots) to explore data patterns.

Model Performance: Classification report, Confusion Matrix, and ROC Curve to assess model accuracy.

Model Interpretation: Feature importance plot showing which factors drive churn.

Retention Strategies: Actionable recommendations based on model insights.

Predict Churn: Input new customer data (single entry or CSV upload) to get real-time churn predictions.

Explainable AI (XAI) Lite: Provides clear explanations for model predictions through feature coefficients.

Non-Engineering Friendly: Designed with clear explanations and interactive elements for easy understanding.

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8+ installed on your system.

Cloning the Repository (If you're getting the project from GitHub)
If you're setting up this project on a new machine or from GitHub, follow these steps first:

Open Terminal/Command Prompt: Open your preferred terminal or command prompt.

Navigate to Desired Directory: Change to the directory where you want to save the project (e.g., cd Documents/GitHub_Projects).

Clone the Repository: Replace your-username with your actual GitHub username.

git clone https://github.com/your-username/ecommerce-churn-dashboard.git

Navigate into Project Folder:

cd ecommerce_churn_dashboard

Continue with Installation Steps: Proceed to "Installation Steps" below, starting from step 3 (Open the Project in VS Code).

Installation Steps
Create a Project Folder:
First, create a new folder on your computer where you'll store all the project files. You can name it something like ecommerce_churn_dashboard.

Download Project Files:
Create the following Python files inside your ecommerce_churn_dashboard folder and copy the content provided in the respective code blocks (from the Gemini chat output) into each file:

data_utils.py

model_utils.py

app_sections.py

dashboard_app.py

requirements.txt (copy the content provided in the requirements.txt immersive block)

Open the Project in VS Code:

Launch VS Code.

Go to File > Open Folder...

Navigate to and select the ecommerce_churn_dashboard folder you just created. Click "Select Folder".

Set Up a Virtual Environment (Highly Recommended):
Using a virtual environment keeps your project's dependencies isolated from other Python projects.

Open the integrated terminal in VS Code: Go to Terminal > New Terminal (or press Ctrl + `  or `).

In the terminal, create a virtual environment:

python -m venv venv

Activate the virtual environment:

On Windows:

.\venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate

You should see (venv) at the beginning of your terminal prompt, indicating that the virtual environment is active.

Install Dependencies:
With your virtual environment activated, install all the required libraries:

In the VS Code terminal, run:

pip install -r requirements.txt

This command reads the requirements.txt file and installs all the specified libraries.

🏃 Running the Dashboard
Once all dependencies are installed, you can run the Streamlit dashboard:

Ensure Virtual Environment is Active: Make sure (venv) is visible in your terminal prompt. If not, activate it again (see step 4 above).

Run the App: In the VS Code terminal, navigate to your ecommerce_churn_dashboard folder (if you're not already there) and run:

streamlit run dashboard_app.py

Access the Dashboard:
After running the command, Streamlit will typically open a new tab in your default web browser pointing to the dashboard (usually http://localhost:8501). If it doesn't open automatically, copy the "Local URL" provided in the terminal and paste it into your browser.

📊 Using the Dashboard
Sidebar Controls: On the left sidebar, you can select your data source (Synthetic, Upload CSV, or Public URL) and configure missing data handling.

Tabs: Navigate through the different tabs at the top of the dashboard to explore the data, model performance, interpretation, and make predictions.

Interactive Elements: Use sliders, select boxes, and buttons to interact with the dashboard and see real-time updates.

⚠️ Data Requirements for Custom CSV
If you upload your own CSV file or use a public URL, ensure it has columns that are similar to the synthetic data. The crucial columns for this dashboard are:

customer_id (unique identifier)

join_date (date customer joined, e.g., YYYY-MM-DD)

last_purchase_date (date of last purchase, e.g., YYYY-MM-DD)

total_purchases (number of purchases)

total_spent (total money spent)

avg_time_on_site_min (average minutes spent on site)

num_support_interactions (number of times customer interacted with support)

product_category_preference (e.g., 'Electronics', 'Apparel', 'Home Goods', 'Books', 'Groceries', 'Beauty')

newsletter_subscribed (Boolean: True/False or numerical: 1/0)

promo_clicks_last_month (number of promotional clicks)

is_churned (Your target variable: 0 for not churned, 1 for churned)

Minor variations in column names might require small adjustments to data_utils.py's engineer_features function.

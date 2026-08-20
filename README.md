# E-commerce Customer Churn Prediction Dashboard

An interactive Streamlit app that predicts which e-commerce customers are likely to churn and explains why, using a Logistic Regression model trained on behavioral and engagement data.

I built this to go beyond a notebook-only ML project — the goal was to wrap a churn model in something a non-technical stakeholder could actually use: upload data, see predictions, and understand the reasoning behind them.

## What it does

- **Flexible input** — works with a built-in synthetic dataset, your own CSV upload, or a public CSV URL, so it isn't tied to one dataset.
- **Feature engineering** — automatically derives Recency, Tenure, Average Purchase Value, and an Engagement Score from raw customer activity fields.
- **Missing data handling** — options to drop or impute (mean/median/mode) before training.
- **Churn model** — Logistic Regression with SMOTE to handle class imbalance, since churned customers are usually a minority class.
- **Evaluation** — classification report, confusion matrix, and ROC curve so you can actually judge how good the model is, not just that it "runs."
- **Model interpretation** — feature coefficients are surfaced as a lightweight explainability layer, showing which factors push a prediction toward churn.
- **Retention suggestions** — the dashboard turns model output into plain-language retention actions instead of leaving it as raw probabilities.
- **Live prediction** — score a single new customer or a batch CSV directly from the UI.

## Tech stack

Python, Streamlit, scikit-learn, imbalanced-learn (SMOTE), pandas.

## Project structure

```
dashboard_app.py       # Streamlit app entry point / page layout
app_sections.py         # Individual dashboard sections (EDA, model perf, prediction, etc.)
data_utils.py            # Data loading, cleaning, feature engineering
model_utils.py         # Model training and evaluation logic
requirements.txt
```

## Running it locally

```bash
git clone https://github.com/Dbansal06/ecommerce-churn-dashboard.git
cd ecommerce-churn-dashboard
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run dashboard_app.py
```

It opens at `http://localhost:8501`.

## Using your own data

If you bring your own CSV, it needs (roughly) these columns:

| Column | Meaning |
|---|---|
| `customer_id` | Unique identifier |
| `join_date` | When the customer signed up |
| `last_purchase_date` | Date of most recent purchase |
| `total_purchases` | Number of purchases to date |
| `total_spent` | Total amount spent |
| `avg_time_on_site_min` | Average session time |
| `num_support_interactions` | Support ticket / contact count |
| `product_category_preference` | e.g. Electronics, Apparel, Books |
| `newsletter_subscribed` | 1/0 or True/False |
| `promo_clicks_last_month` | Promo engagement |
| `is_churned` | Target label (0 = active, 1 = churned) |

If your column names differ slightly, you'll need to tweak `engineer_features()` in `data_utils.py`.

## What I'd improve next

- Swap Logistic Regression for a gradient-boosted model and compare performance.
- Add proper cross-validation instead of a single train/test split.
- Persist trained models instead of retraining on every run.

## License

MIT

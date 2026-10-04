# RetailPulse (core version)

Flask + scikit-learn. Sales & profit (Random Forest Regressor) and churn (Random Forest Classifier).
The bundled data is generated **DEMO DATA**, not a real company.

## Setup (Windows)
1. Install Python 3.10+ from python.org (tick "Add to PATH").
2. `python -m venv venv`
3. `venv\Scripts\activate`
4. `pip install -r requirements.txt`
5. Dataset: run step 6 to auto-create `data/retail_data.csv`, or replace it with your own CSV using the same columns:
   customer_id, customer_age, gender, region, product_category, product, order_date, quantity, sales, cost, profit,
   discount, total_orders, total_spending, average_order_value, days_since_last_purchase, complaints,
   customer_tenure, churn, discount_usage
6. `python ml/train.py` (cleans data, trains, saves models and real metrics)
7. `python app.py`
8. Open http://127.0.0.1:5000

## Testing
Use each sidebar page, press the predict button, and check the result card. Invalid input (negative/empty) shows an error. Predictions are logged in `database/retailpulse.db`.

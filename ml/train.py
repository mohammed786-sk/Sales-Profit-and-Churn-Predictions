"""Run once: python ml/train.py
Creates DEMO data (if data/retail_data.csv is missing), cleans it, trains the
3 Random Forest models and saves them + REAL test metrics to models/."""
import os, json, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "retail_data.csv")
CATS = ["Electronics", "Clothing", "Grocery", "Furniture", "Beauty"]
REGIONS = ["North", "South", "East", "West"]
PRICE = dict(zip(CATS, [1200, 500, 150, 2000, 300]))


def make_demo():
    """DEMO DATA ONLY - randomly generated, does not represent a real company."""
    r = np.random.default_rng(42); nc, n = 800, 4000
    c = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(nc)],
        "customer_age": r.integers(18, 70, nc), "gender": r.choice(["Male", "Female"], nc),
        "region": r.choice(REGIONS, nc), "customer_tenure": r.integers(1, 60, nc),
        "complaints": r.poisson(1, nc), "days_since_last_purchase": r.integers(1, 365, nc)})
    z = .015 * c.days_since_last_purchase + .5 * c.complaints - .03 * c.customer_tenure - 2.5
    c["churn"] = (r.random(nc) < 1 / (1 + np.exp(-z))).astype(int)
    o = pd.DataFrame({"customer_id": r.choice(c.customer_id, n), "product_category": r.choice(CATS, n),
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(r.integers(0, 730, n), unit="D"),
        "quantity": r.integers(1, 10, n), "discount": r.choice([0, .05, .1, .2], n)})
    o["product"] = o.product_category + " " + r.integers(1, 6, n).astype(str)
    o["sales"] = (o.product_category.map(PRICE) * o.quantity * (1 - o.discount) * r.uniform(.9, 1.1, n)).round(2)
    o["cost"] = (o.sales * r.uniform(.55, .9, n)).round(2)
    o["profit"] = (o.sales - o.cost).round(2)
    g = o.groupby("customer_id").agg(total_orders=("sales", "size"), total_spending=("sales", "sum"),
                                     discount_usage=("discount", "mean")).reset_index()
    g["average_order_value"] = g.total_spending / g.total_orders
    d = o.merge(c, on="customer_id").merge(g, on="customer_id")
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    d.to_csv(CSV, index=False)


def clean(df):
    """Duplicates, missing values, types."""
    df = df.drop_duplicates().copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in df.select_dtypes("number"):
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["order_date"])
    df["month"] = df.order_date.dt.month
    df["discount_usage"] = df.get("discount_usage", df.discount)
    return df


def pipe(cat, model):
    """One-hot encode text columns, pass numbers through, then the model."""
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    return Pipeline([("pre", pre), ("m", model)])


def reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return dict(MAE=mean_absolute_error(y, p), MSE=mse, RMSE=mse ** .5, R2=r2_score(y, p))


def main():
    if not os.path.exists(CSV): make_demo()
    df = clean(pd.read_csv(CSV))
    # Previous sales/profit = a noisy lag proxy (demo data has no history column)
    rg = np.random.default_rng(1)
    df["prev_sales"] = df.sales * rg.uniform(.8, 1.2, len(df))
    df["prev_profit"] = df.profit * rg.uniform(.8, 1.2, len(df))
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    out = {}
    for name, cat, num, tgt in [
        ("sales", ["product_category", "region"], ["prev_sales", "quantity", "month", "discount"], "sales"),
        ("profit", ["product_category", "region"], ["sales", "cost", "discount", "quantity", "prev_profit"], "profit")]:
        X, y = df[cat + num], df[tgt]
        Xt, Xs, yt, ys = train_test_split(X, y, test_size=.2, random_state=42)
        m = pipe(cat, RandomForestRegressor(100, min_samples_leaf=5, random_state=42, n_jobs=-1)).fit(Xt, yt)
        out[name] = reg_metrics(ys, m.predict(Xs))
        joblib.dump(m, os.path.join(BASE, "models", f"{name}_model.pkl"), compress=3)
    cust = df.drop_duplicates("customer_id")
    cat = ["gender"]
    num = ["customer_age", "total_orders", "total_spending", "average_order_value",
           "days_since_last_purchase", "complaints", "customer_tenure", "discount_usage"]
    Xt, Xs, yt, ys = train_test_split(cust[cat + num], cust.churn, test_size=.2, random_state=42, stratify=cust.churn)
    m = pipe(cat, RandomForestClassifier(100, min_samples_leaf=3, random_state=42, class_weight="balanced")).fit(Xt, yt)
    p = m.predict(Xs)
    out["churn"] = dict(Accuracy=accuracy_score(ys, p), Precision=precision_score(ys, p, zero_division=0),
        Recall=recall_score(ys, p, zero_division=0), F1=f1_score(ys, p, zero_division=0),
        confusion=confusion_matrix(ys, p).tolist())
    joblib.dump(m, os.path.join(BASE, "models", "churn_model.pkl"), compress=3)
    json.dump(out, open(os.path.join(BASE, "models", "metrics.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

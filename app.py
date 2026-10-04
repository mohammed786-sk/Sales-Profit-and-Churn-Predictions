import os, json, sqlite3, joblib
import pandas as pd
from werkzeug.exceptions import HTTPException
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
# Vercel's disk is read-only except /tmp, so the log DB goes there when deployed.
DB = "/tmp/retailpulse.db" if os.environ.get("VERCEL") else os.path.join(BASE, "database", "retailpulse.db")


def log_prediction(kind, row, res):
    """Save the prediction in SQLite. Never let a logging problem break the app."""
    try:
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        with sqlite3.connect(DB) as c:
            c.execute("CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, kind TEXT, input TEXT, result TEXT, at TEXT DEFAULT CURRENT_TIMESTAMP)")
            c.execute("INSERT INTO predictions(kind,input,result) VALUES(?,?,?)", (kind, json.dumps(row), json.dumps(res)))
    except Exception as e:
        print("Log skipped:", e)

# Load data and models safely: a missing file must never crash the whole site.
df, models, LOAD_ERROR = None, {}, ""
try:
    df = pd.read_csv(os.path.join(BASE, "data", "retail_data.csv"), parse_dates=["order_date"])
    df["month"] = df.order_date.dt.to_period("M").astype(str)
    models = {k: joblib.load(os.path.join(BASE, "models", f"{k}_model.pkl")) for k in ("sales", "profit", "churn")}
except Exception as e:
    LOAD_ERROR = f"Data/model files could not be loaded ({e}). Run: python ml/train.py"
    print(LOAD_ERROR)


@app.errorhandler(Exception)
def any_error(e):
    """Pages keep the normal error page; API/prediction calls ALWAYS get JSON."""
    code = e.code if isinstance(e, HTTPException) else 500
    if request.path.startswith(("/predict", "/api")):
        return jsonify(error=getattr(e, "description", None) if isinstance(e, HTTPException) else f"Server error: {e}"), code
    return (e if isinstance(e, HTTPException) else ("Server error", 500))

# (text fields, number fields) for each model - must match ml/train.py
SPEC = {"sales": (["product_category", "region"], ["prev_sales", "quantity", "month", "discount"]),
        "profit": (["product_category", "region"], ["sales", "cost", "discount", "quantity", "prev_profit"]),
        "churn": (["gender"], ["customer_age", "total_orders", "total_spending", "average_order_value",
                               "days_since_last_purchase", "complaints", "customer_tenure", "discount_usage"])}


@app.route("/")
@app.route("/dashboard")
def index():
    return render_template("index.html")


@app.post("/predict-<kind>")
def predict(kind):
    if kind not in SPEC:
        return jsonify(error="Unknown model"), 404
    if kind not in models:
        return jsonify(error=LOAD_ERROR or "Model not loaded"), 503
    data = request.get_json(silent=True) or {}
    cat, num = SPEC[kind]
    row = {}
    try:
        for f in cat:
            if not str(data.get(f, "")).strip(): raise ValueError(f"{f} is required")
            row[f] = str(data[f])
        for f in num:
            v = float(data[f])
            if v < 0: raise ValueError(f"{f} cannot be negative")
            row[f] = v
    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400
    X = pd.DataFrame([row])
    try:
        if kind == "churn":
            p = float(models[kind].predict_proba(X)[0][1])
            res = {"probability": round(p * 100, 1), "status": "LIKELY TO CHURN" if p >= .5 else "NOT LIKELY TO CHURN"}
        else:
            val = float(models[kind].predict(X)[0])
            res = {"prediction": round(val, 2)}
            if kind == "profit":
                m = val / row["sales"] * 100 if row["sales"] else 0
                res.update(margin=round(m, 1), level="High Profit" if m >= 30 else "Medium Profit" if m >= 15 else "Low Profit")
        log_prediction(kind, row, res)
        return jsonify(res)
    except Exception as e:  # always answer with JSON, never an HTML error page
        return jsonify(error=f"Prediction failed: {e}"), 500


def grp(col, val):
    s = df.groupby(col)[val].sum().round(0)
    return {"labels": list(map(str, s.index)), "values": s.tolist()}


@app.get("/api/dashboard")
def dashboard():
    if df is None:
        return jsonify(error=LOAD_ERROR), 503
    cust = df.drop_duplicates("customer_id")
    mon = df.groupby("month").agg(sales=("sales", "sum"), profit=("profit", "sum"), orders=("sales", "size")).round(0)
    top = df.groupby("customer_id").sales.sum().nlargest(10).round(0)
    cs, rp = df.groupby("product_category").sales.sum(), df.groupby("region").profit.sum()
    ins = [f"{cs.idxmax()} generated the highest sales (₹{cs.max():,.0f}).",
           f"{rp.idxmax()} region generated the highest profit (₹{rp.max():,.0f}).",
           f"Overall churn rate is {cust.churn.mean() * 100:.1f}% of {len(cust)} customers."]
    a, b = cust[cust.churn == 1].days_since_last_purchase.mean(), cust[cust.churn == 0].days_since_last_purchase.mean()
    ins.append(f"Churned customers waited {a:.0f} days since last purchase on average vs {b:.0f} for active ones.")
    half = len(mon) // 2
    if half:
        ch = (mon.sales.iloc[half:].sum() / mon.sales.iloc[:half].sum() - 1) * 100
        ins.append(f"Sales in the later half of the period were {abs(ch):.1f}% {'higher' if ch >= 0 else 'lower'} than the earlier half.")
    return jsonify(
        kpi={"sales": float(df.sales.sum()), "profit": float(df.profit.sum()), "customers": len(cust),
             "orders": len(df), "aov": float(df.sales.mean()), "churn": float(cust.churn.mean() * 100)},
        monthly={"labels": list(mon.index), "sales": mon.sales.tolist(), "profit": mon.profit.tolist(), "orders": mon.orders.tolist()},
        category_sales=grp("product_category", "sales"), category_profit=grp("product_category", "profit"),
        region_sales=grp("region", "sales"),
        churn=[int((cust.churn == 1).sum()), int((cust.churn == 0).sum())],
        top_customers={"labels": top.index.tolist(), "values": top.tolist()}, insights=ins)


@app.get("/api/metrics")
def metrics():
    return jsonify(json.load(open(os.path.join(BASE, "models", "metrics.json"))))


if __name__ == "__main__":
    app.run(debug=True)

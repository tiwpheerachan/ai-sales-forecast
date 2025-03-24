import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📊 Advanced Sales & Product Forecasting", layout="wide")

@st.cache_data
def load_excel(file):
    xls = pd.ExcelFile(file)
    dfs = {}
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()
            dfs[sheet] = df
        except:
            continue
    return dfs

@st.cache_resource
def train_model(df_perf, df_gmv):
    # ... (โค้ดทำความสะอาดข้อมูลเดิม) ...

    # Calculate Growth Rate and Trend
    df["month_numeric"] = df["year_month"].apply(lambda x: int(x.replace("-", "")))
    growth_rates = df.groupby(["product_name", "platform"]).apply(
        lambda g: g.sort_values("month_numeric").assign(
            pct_change=g["sales_thb"].pct_change().fillna(0)
        )["pct_change"].mean()
    ).reset_index(name="avg_growth_rate")

    def get_trend(series):
        try:
            decomposition = seasonal_decompose(series, model='additive', extrapolate_trend='freq')
            return decomposition.trend.iloc[-1]
        except:
            return 0

    trend_data = df.groupby("year_month")["sales_thb"].sum().reset_index()
    trend_data["trend"] = trend_data["sales_thb"].rolling(window=3).apply(get_trend, raw=False)
    trend_map = dict(zip(trend_data["year_month"], trend_data["trend"]))
    df["trend"] = df["year_month"].map(trend_map).fillna(0)

    # Summary
    summary = df.groupby(["brand", "product_name", "platform", "year_month", "campaign_type"]).agg({
        "sales_thb": "sum", "units_sold": "sum", "conversion_rate": "mean", "trend": "mean"
    }).reset_index()
    summary = pd.merge(summary, growth_rates, on=["product_name", "platform"], how="left").fillna(0)

    # Encode
    le_brand, le_product, le_platform, le_campaign = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
    summary["month_enc"] = summary["year_month"].apply(lambda x: int(x.replace("-", "")))
    summary["brand_enc"] = le_brand.fit_transform(summary["brand"])
    summary["product_enc"] = le_product.fit_transform(summary["product_name"])
    summary["platform_enc"] = le_platform.fit_transform(summary["platform"])
    summary["campaign_enc"] = le_campaign.fit_transform(summary["campaign_type"])

    # Train Model (ปลอด NaN และ Infinity)
    X = summary[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate", "trend"]]
    y = summary["sales_thb"]

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]

    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6]
    }
    model = GradientBoostingRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    encoders = {"brand": le_brand, "product": le_product, "platform": le_platform, "campaign": le_campaign}
    return best_model, summary, encoders

def forecast_future(summary, model, encoders, months_ahead, growth_expectation=1.0):
    # ... (โค้ดทำนายอนาคตเดิม) ...

    # Merge avg_growth_rate และ trend
    growth_lookup = summary.groupby(["product_name", "platform"])["avg_growth_rate"].mean().reset_index()
    future = pd.merge(future, growth_lookup, on=["product_name", "platform"], how="left")
    future["avg_growth_rate"] = future["avg_growth_rate"].fillna(0) * growth_expectation

    trend_lookup = summary.groupby("year_month")["trend"].mean().reset_index()
    future["trend"] = future["year_month"].map(dict(zip(trend_lookup["year_month"], trend_lookup["trend"]))).fillna(0)

    # Features ให้ตรงกับตอน train
    X = future[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate", "trend"]]

    # ทำนาย
    future["forecast_sales"] = model.predict(X)

    return future

def recommend_insights(df):
    # ... (โค้ดแนะนำเดิม) ...
    return msg

# === Streamlit UI ===
# ... (โค้ด UI เดิม) ...

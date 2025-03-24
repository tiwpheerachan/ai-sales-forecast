import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="📊 AI Sales & Product Forecasting", layout="wide")

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
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")

    def classify_campaign_type(date):
        if pd.isna(date): return "unknown"
        if date.day == date.month: return "dday"
        elif date.day == 15: return "midmonth"
        elif date.day == 25: return "payday"
        else: return "normal_day"

    df_gmv["campaign_type"] = df_gmv["Data"].apply(classify_campaign_type)
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M")

    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

    df_perf["month_num"] = df_perf["Month"].map(month_map)
    df_perf["date"] = pd.to_datetime(dict(year=df_perf["Year"], month=df_perf["month_num"], day=1))
    df_perf["year_month"] = df_perf["date"].dt.to_period("M")
    campaign_map = df_gmv[["year_month", "campaign_type"]].drop_duplicates()
    df = pd.merge(df_perf, campaign_map, on="year_month", how="left")

    df = df.rename(columns={
        "Product": "product_name",
        "Brand": "brand",
        "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb",
        "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    summary = df.groupby(["brand", "product_name", "platform", "year_month", "campaign_type"]).agg({
        "sales_thb": "sum",
        "units_sold": "sum",
        "conversion_rate": "mean"
    }).reset_index()

    le = LabelEncoder()
    summary["brand_enc"] = le.fit_transform(summary["brand"])
    summary["product_enc"] = le.fit_transform(summary["product_name"])
    summary["platform_enc"] = le.fit_transform(summary["platform"])
    summary["campaign_enc"] = le.fit_transform(summary["campaign_type"])
    summary["month_enc"] = le.fit_transform(summary["year_month"])

    X = summary[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc"]]
    y = summary["sales_thb"]

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=5) # Adjusted parameters
    model.fit(X, y)
    return model, summary, le

def future_data(df, months_ahead, le):
    future_months = pd.date_range(datetime.today(), periods=months_ahead, freq='MS').to_period("M").astype(str)
    future_rows = []
    unique = df[["brand", "product_name", "platform"]].drop_duplicates() # Removed campaign_type
    for month in future_months:
        for _, row in unique.iterrows():
            future_rows.append({
                "brand": row["brand"],
                "product_name": row["product_name"],
                "platform": row["platform"],
                "campaign_type": "normal_day", # Set to normal_day to reduce variance
                "year_month": month
            })
    future_df = pd.DataFrame(future_rows)
    future_df["brand_enc"] = le.fit_transform(future_df["brand"])
    future_df["product_enc"] = le.fit_transform(future_df["product_name"])
    future_df["platform_enc"] = le.fit_transform(future_df["platform"])
    future_df["campaign_enc"] = le.fit_transform(future_df["campaign_type"])
    future_df["month_enc"] = le.fit_transform(future_df["year_month"])
    return future_df

# === UI ===
st.title("🧠 AI Sales & Product Forecasting Dashboard")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
months_to_predict = st.sidebar.slider("🔮 Forecast Months Ahead", 1, 60, 3)

if uploaded_file:
    data = load_excel(uploaded_file)
    perf_sheet = next((k for k in data if "Perf" in k or "Performance" in k), None)
    gmv_sheet = next((k for k in data if "GMV" in k), None)

    if perf_sheet and gmv_sheet:
        df_perf = data[perf_sheet]
        df_gmv = data[gmv_sheet]
        model, df_raw, le = train_model(df_perf, df_gmv)
        df_future = future_data(df_raw, months_to_predict, le)

        X_future = df_future[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc"]]
        df_future["forecast_sales"] = model.predict(X_future)

        st.sidebar.subheader("🎯 Filter")
        selected_platform = st.sidebar.selectbox("เลือกแพลตฟอร์ม", ["All"] + sorted(df_future["platform"].dropna().unique()))
        if selected_platform != "All":
            df_future = df_future[df_future["platform"] == selected_platform]

        # === Metrics ===
        total_sales = df_future["forecast_sales"].sum()
        total_skus = df_future["product_name"].nunique()
        avg_sales = total_sales / total_skus if total_skus > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Forecasted Sales", f"{total_sales:,.2f} THB")
        col2.metric("📦 Total SKUs", f"{total_skus:,}")
        col3.metric("📈 Avg. Sales per SKU", f"{avg_sales:,.2f}")

        # === Forecast Graphs ===
        st.subheader("📊 Forecast Trend by Month")
        df_

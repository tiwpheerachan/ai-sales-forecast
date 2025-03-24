
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📊 AI Sales & Product Forecasting Dashboard", layout="wide")

@st.cache_data
def load_excel(file):
    xls = pd.ExcelFile(file)
    sheets = xls.sheet_names
    dfs = {}
    for sheet in sheets:
        try:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()
            dfs[sheet] = df
        except:
            continue
    return dfs

@st.cache_resource
def prepare_model(df, date_col):
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    summary = df.groupby(["platform", "product_name", "year_month"]).agg({
        "sales_thb": "sum"
    }).reset_index()

    le = LabelEncoder()
    summary["product_enc"] = le.fit_transform(summary["product_name"])
    summary["platform_enc"] = le.fit_transform(summary["platform"])
    summary["month_enc"] = le.fit_transform(summary["year_month"])

    X = summary[["product_enc", "platform_enc", "month_enc"]]
    y = summary["sales_thb"]
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model, summary, le

@st.cache_data
def forecast(model, summary, le, months_to_forecast=3):
    future_months = sorted(summary["year_month"].unique())[-1:]
    future_encoded = le.transform(summary["product_name"])
    forecast_data = summary.copy()
    forecast_data["forecast_sales"] = model.predict(forecast_data[["product_enc", "platform_enc", "month_enc"]])
    return forecast_data

st.sidebar.header("📁 Upload Excel")
uploaded_file = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])

if uploaded_file:
    dfs = load_excel(uploaded_file)
    perf_sheet = [k for k in dfs if "perf" in k.lower()][0]
    df = dfs[perf_sheet]
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Product": "product_name",
        "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb",
        "Month": "month",
        "Year": "year"
    })
    df["month_num"] = df["month"].str[:3].str.upper().map({
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    })
    df["day"] = 1
    df["order_date"] = pd.to_datetime(dict(year=df["year"], month=df["month_num"], day=df["day"]))

    model, summary, le = prepare_model(df, "order_date")
    forecast_df = forecast(model, summary, le)

    st.title("📊 AI Sales & Product Forecasting Dashboard")
    platform_filter = st.sidebar.selectbox("Platform", ["All"] + sorted(forecast_df["platform"].unique()))
    if platform_filter != "All":
        forecast_df = forecast_df[forecast_df["platform"] == platform_filter]

    total_sales = forecast_df["forecast_sales"].sum()
    total_skus = forecast_df["product_name"].nunique()
    avg_sales = forecast_df["forecast_sales"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔮 Total Forecast Sales", f"{total_sales:,.0f} THB")
    col2.metric("📦 Total SKUs", total_skus)
    col3.metric("📈 Avg/SKU", f"{avg_sales:,.2f} THB")

    fig = px.bar(forecast_df.sort_values("forecast_sales", ascending=False).head(30),
                 x="forecast_sales", y="product_name", color="platform", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Forecasted Product Table")
    st.dataframe(forecast_df.sort_values("forecast_sales", ascending=False), use_container_width=True)
else:
    st.info("📤 Please upload an Excel file to start.")

%%writefile app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="📊 AI Sales Forecast: Shopee + Lazada", layout="wide")

@st.cache_data
def load_data(file):
    shopee_df = pd.read_excel(file, sheet_name="Shopee_Product_Perf")
    lazada_df = pd.read_excel(file, sheet_name="Lazada_Product_Perf")
    gmv_df = pd.read_excel(file, sheet_name="GMV_DATA")

    gmv_df["Data"] = pd.to_datetime(gmv_df["Data"], errors="coerce")

    def classify_campaign_type(date):
        if pd.isna(date): return "unknown"
        if date.day == date.month: return "dday"
        elif date.day == 15: return "midmonth"
        elif date.day == 25: return "payday"
        else: return "normal_day"

    gmv_df["campaign_type"] = gmv_df["Data"].apply(classify_campaign_type)
    gmv_df["year_month"] = gmv_df["Data"].dt.to_period("M")

    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

    # Shopee
    shopee_df["month_num"] = shopee_df["Month"].map(month_map)
    shopee_df["date"] = pd.to_datetime(dict(year=shopee_df["Year"], month=shopee_df["month_num"], day=1))
    shopee_df["year_month"] = shopee_df["date"].dt.to_period("M")
    shopee_df["platform"] = "Shopee"
    shopee_df = pd.merge(shopee_df, gmv_df[["year_month", "campaign_type"]].drop_duplicates(), on="year_month", how="left")
    shopee_df = shopee_df.rename(columns={
        "Product": "product_name",
        "Brand": "brand",
        "Sales (Confirmed Order) (THB)": "sales_thb",
        "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })

    # Lazada
    lazada_df["month_num"] = lazada_df["Month"].map(month_map)
    lazada_df["date"] = pd.to_datetime(dict(year=lazada_df["Year"], month=lazada_df["month_num"], day=1))
    lazada_df["year_month"] = lazada_df["date"].dt.to_period("M")
    lazada_df["platform"] = "Lazada"
    lazada_df = pd.merge(lazada_df, gmv_df[["year_month", "campaign_type"]].drop_duplicates(), on="year_month", how="left")
    lazada_df = lazada_df.rename(columns={
        "Product Name": "product_name",
        "Brand": "brand",
        "Revenue": "sales_thb",
        "Units Sold": "units_sold",
        "Conversion Rate": "conversion_rate"
    })

    cols_needed = ["product_name", "brand", "platform", "date", "sales_thb", "units_sold", "conversion_rate", "campaign_type"]
    df = pd.concat([shopee_df[cols_needed], lazada_df[cols_needed]], ignore_index=True)
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df

@st.cache_resource
def train_model(df):
    df = df[df["campaign_type"].isin(["dday", "midmonth", "payday"])].copy()

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, summary, le

def forecast(model, summary, le, month, campaign, platform_filter=None, brand_filter=None, sku_keyword=None):
    input_data = summary.copy()
    input_data = input_data[input_data["campaign_type"] == campaign]
    input_data["month_enc"] = le.fit_transform([month] * len(input_data))

    if platform_filter:
        input_data = input_data[input_data["platform"] == platform_filter]
    if brand_filter:
        input_data = input_data[input_data["brand"] == brand_filter]
    if sku_keyword:
        input_data = input_data[input_data["product_name"].str.contains(sku_keyword, case=False, na=False)]

    X_new = input_data[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc"]]
    y_pred = model.predict(X_new)
    input_data["forecast_sales"] = y_pred
    return input_data

# === UI START ===
st.sidebar.header("📂 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Upload .xlsx file", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    model, summary, le = train_model(df)

    st.sidebar.subheader("🔎 Filters")
    month_options = sorted(df["year_month"].unique())
    selected_month = st.sidebar.selectbox("Select Month", month_options)
    campaign_type = st.sidebar.selectbox("Select Campaign", ["dday", "midmonth", "payday"])
    platform_options = sorted(df["platform"].dropna().unique())
    selected_platform = st.sidebar.selectbox("Select Platform", ["All"] + platform_options)
    brand_options = sorted(df["brand"].dropna().unique())
    selected_brand = st.sidebar.selectbox("Select Brand", ["All"] + brand_options)
    sku_input = st.sidebar.text_input("Search SKU Keyword")

    platform_filter = None if selected_platform == "All" else selected_platform
    brand_filter = None if selected_brand == "All" else selected_brand

    forecast_df = forecast(model, summary, le, selected_month, campaign_type, platform_filter, brand_filter, sku_input)

    st.title("📊 Shopee + Lazada: Sales Forecast Dashboard")
    st.markdown(f"### Month: `{selected_month}` | Campaign: `{campaign_type}` | Platform: `{selected_platform}`")

    total_sales = forecast_df["forecast_sales"].sum()
    total_skus = len(forecast_df)
    avg_sales = total_sales / total_skus if total_skus > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Forecasted Sales", f"{total_sales:,.0f} THB")
    col2.metric("📦 Total SKUs", f"{total_skus}")
    col3.metric("📈 Avg. per SKU", f"{avg_sales:,.2f} THB")

    if not forecast_df.empty:
        fig1 = px.bar(forecast_df.sort_values("forecast_sales", ascending=True), 
                    x="forecast_sales", y="product_name", orientation="h",
                    title="🏆 Forecasted Sales by SKU", color="forecast_sales")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.pie(forecast_df, names="platform", values="forecast_sales", hole=0.4,
                    title="🛒 Forecast Share by Platform")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📄 Forecasted Results")
    st.dataframe(forecast_df, use_container_width=True)


import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="📊 AI Sales Forecast Dashboard", layout="wide")

@st.cache_data
def load_data(file):
    perf_df = pd.read_excel(file, sheet_name="Shopee_Product_Perf")
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

    perf_df["month_num"] = perf_df["Month"].map(month_map)
    perf_df["date"] = pd.to_datetime(dict(year=perf_df["Year"], month=perf_df["month_num"], day=1))
    perf_df["year_month"] = perf_df["date"].dt.to_period("M")

    campaign_map = gmv_df[["year_month", "campaign_type"]].drop_duplicates()
    df = pd.merge(perf_df, campaign_map, on="year_month", how="left")

    df = df[["Product", "Brand", "Platforms", "date", 
             "Sales (Confirmed Order) (THB)", "Units (Confirmed Order)",
             "Conversion Rate (Confirmed Order)", "campaign_type"]].copy()

    df.columns = ["product_name", "brand", "platform", "date", 
                  "sales_thb", "units_sold", "conversion_rate", "campaign_type"]
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    return df

@st.cache_resource
def train_model(df):
    df = df[df["campaign_type"].isin(["dday", "midmonth", "payday"])]
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

def forecast(model, summary, le, month, campaign, brand_filter=None, sku_keyword=None):
    input_data = summary.copy()
    input_data = input_data[input_data["campaign_type"] == campaign]
    input_data["month_enc"] = le.fit_transform([month] * len(input_data))

    if brand_filter:
        input_data = input_data[input_data["brand"] == brand_filter]
    if sku_keyword:
        input_data = input_data[input_data["product_name"].str.contains(sku_keyword, case=False, na=False)]

    X_new = input_data[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc"]]
    y_pred = model.predict(X_new)

    input_data["forecast_sales"] = y_pred
    return input_data[["brand", "product_name", "platform", "year_month", "campaign_type", "forecast_sales"]]

st.sidebar.header("📂 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ Excel", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    model, summary, le = train_model(df)

    st.sidebar.subheader("🔍 Filters")
    all_months = sorted(df["year_month"].unique())
    selected_month = st.sidebar.selectbox("Select Month", all_months)
    campaign_type = st.sidebar.selectbox("Select Campaign", ["dday", "midmonth", "payday"])
    brand_options = sorted(df["brand"].dropna().unique())
    selected_brand = st.sidebar.selectbox("Select Brand", ["All"] + brand_options)
    sku_input = st.sidebar.text_input("Search SKU Keyword")

    brand_filter = None if selected_brand == "All" else selected_brand
    forecast_df = forecast(model, summary, le, selected_month, campaign_type, brand_filter, sku_input)

    st.title("📊 Sales & Product Forecast Dashboard")
    st.markdown(f"### Forecast for `{selected_month}` | Campaign: `{campaign_type}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Forecasted Sales", f"{forecast_df['forecast_sales'].sum():,.2f}")
    col2.metric("📦 SKUs", f"{len(forecast_df)}")
    col3.metric("📈 Avg. Sales/SKU", f"{forecast_df['forecast_sales'].mean():,.2f}")

    st.plotly_chart(px.bar(forecast_df.sort_values("forecast_sales"), x="forecast_sales", y="product_name", orientation="h"), use_container_width=True)
    st.plotly_chart(px.pie(forecast_df, names="platform", values="forecast_sales", title="🛒 Platform Share", hole=0.4), use_container_width=True)

    st.subheader("🗂 Forecast Table")
    st.dataframe(forecast_df, use_container_width=True)

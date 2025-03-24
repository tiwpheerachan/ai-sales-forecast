
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="📊 Sales Forecast Dashboard", layout="wide")

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
def prepare_and_forecast(df_perf, df_gmv):
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
    model = GradientBoostingRegressor()
    model.fit(X, y)
    summary["forecast_sales"] = model.predict(X)

    return summary

# === UI ===
st.sidebar.title("📂 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Upload your .xlsx file", type=["xlsx"])

if uploaded_file:
    sheets = load_excel(uploaded_file)
    perf_sheets = [k for k in sheets if "Perf" in k or "Performance" in k]
    gmv_sheets = [k for k in sheets if "GMV" in k]

    if perf_sheets and gmv_sheets:
        df_perf = sheets[perf_sheets[0]]
        df_gmv = sheets[gmv_sheets[0]]

        summary = prepare_and_forecast(df_perf, df_gmv)

        st.sidebar.subheader("🔎 Filter")
        month = st.sidebar.selectbox("Select Month", sorted(summary["year_month"].unique()))
        campaign = st.sidebar.selectbox("Select Campaign", summary["campaign_type"].unique())
        platform = st.sidebar.selectbox("Select Platform", ["All"] + sorted(summary["platform"].unique()))

        df_filtered = summary[(summary["year_month"] == month) & (summary["campaign_type"] == campaign)]
        if platform != "All":
            df_filtered = df_filtered[df_filtered["platform"] == platform]

        st.title("📊 Shopee + Lazada: Sales Forecast Dashboard")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Forecasted Sales", f"{df_filtered['forecast_sales'].sum():,.0f} THB")
        col2.metric("📦 Total SKUs", len(df_filtered))
        col3.metric("📈 Avg. per SKU", f"{df_filtered['forecast_sales'].mean():,.2f} THB")

        st.subheader("📌 Forecasted Sales by SKU")
        fig = px.bar(df_filtered.sort_values("forecast_sales", ascending=False).head(50),
                     x="forecast_sales", y="product_name", orientation="h", 
                     color="forecast_sales", height=800)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📄 Forecast Data")
        st.dataframe(df_filtered, use_container_width=True)

    else:
        st.error("❗ Cannot find appropriate sheets for Performance and GMV. Please upload a valid file.")
else:
    st.info("📤 Please upload a .xlsx file to get started.")

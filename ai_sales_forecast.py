import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta
from datetime import datetime

st.set_page_config(page_title="📊 Sales Forecast Extended", layout="wide")

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

def classify_campaign_type(date):
    if pd.isna(date): return "unknown"
    if date.day == date.month: return "dday"
    elif date.day == 15: return "midmonth"
    elif date.day == 25: return "payday"
    else: return "normal_day"

@st.cache_resource
def train_forecast_model(df_perf, df_gmv):
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
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
    for col in ["brand", "product_name", "platform", "campaign_type", "year_month"]:
        summary[col + "_enc"] = le.fit_transform(summary[col])

    X = summary[[c for c in summary.columns if "_enc" in c]]
    y = summary["sales_thb"]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    return model, summary, le

def forecast_future(model, summary, le, months_ahead):
    future = []
    base_month = datetime(2025, 3, 24)
    for i in range(1, months_ahead + 1):
        month = (base_month + relativedelta(months=i)).strftime("%Y-%m")
        for _, row in summary.iterrows():
            data = row.copy()
            data["year_month"] = month
            for col in ["brand", "product_name", "platform", "campaign_type", "year_month"]:
                data[col + "_enc"] = le.fit_transform(summary[col])[0]  # simplified fit
            future.append(data)

    future_df = pd.DataFrame(future)
    X_new = future_df[[c for c in future_df.columns if "_enc" in c]]
    future_df["forecast_sales"] = model.predict(X_new)
    return future_df

def ai_recommendation(df):
    top = df.groupby("campaign_type")["forecast_sales"].sum().idxmax()
    return f"🎯 ควรโปรโมตในช่วง **{top}** เพราะมีแนวโน้มยอดขายสูงที่สุดจากผลการทำนาย"

# ========== UI ==========
st.sidebar.header("📁 Upload Excel")
uploaded_file = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])
months_select = st.sidebar.slider("🔮 ทำนายล่วงหน้า (เดือน)", 1, 60, 6)

if uploaded_file:
    sheets = load_excel(uploaded_file)
    perf_sheets = [k for k in sheets if "Perf" in k or "Performance" in k]
    gmv_sheets = [k for k in sheets if "GMV" in k]

    if perf_sheets and gmv_sheets:
        df_perf = sheets[perf_sheets[0]]
        df_gmv = sheets[gmv_sheets[0]]
        model, summary, le = train_forecast_model(df_perf, df_gmv)

        forecast_df = forecast_future(model, summary, le, months_select)
        forecast_df["year_month"] = forecast_df["year_month"].astype(str)

        # --- Dashboard Cards ---
        total_sales = forecast_df["forecast_sales"].sum()
        unique_skus = forecast_df["product_name"].nunique()
        st.title("📊 Extended Sales Forecast Dashboard")

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Forecasted Sales", f"{total_sales:,.0f} THB")
        c2.metric("📦 Unique SKUs", unique_skus)
        c3.metric("📆 Months Forecasted", months_select)

        # --- AI Recommendation ---
        st.subheader("💡 AI แนะนำช่วงที่ควรโปรโมต")
        st.success(ai_recommendation(forecast_df))

        # --- Trend Line ---
        trend = forecast_df.groupby("year_month")["forecast_sales"].sum().reset_index()
        fig1 = px.line(trend, x="year_month", y="forecast_sales", markers=True, title="📈 แนวโน้มยอดขายรวม")
        st.plotly_chart(fig1, use_container_width=True)

        # --- Top Products ---
        top_sku = forecast_df.groupby("product_name")["forecast_sales"].sum().nlargest(10).reset_index()
        fig2 = px.bar(top_sku, x="forecast_sales", y="product_name", orientation="h", title="🏆 Top 10 สินค้าขายดี")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📄 รายละเอียดการทำนายทั้งหมด")
        st.dataframe(forecast_df, use_container_width=True)
    else:
        st.error("❗ ไม่พบข้อมูล Performance หรือ GMV ที่เหมาะสม")
else:
    st.info("📥 กรุณาอัปโหลดไฟล์ Excel เพื่อเริ่มต้นการใช้งาน")

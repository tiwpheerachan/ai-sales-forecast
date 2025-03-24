import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="📊 AI Forecast Extended", layout="wide")

# === LOAD DATA ===
@st.cache_data
def load_excel(file):
    xls = pd.ExcelFile(file)
    sheets = xls.sheet_names
    data = {}
    for s in sheets:
        try:
            df = xls.parse(s)
            df.columns = df.columns.str.strip()
            data[s] = df
        except:
            continue
    return data

# === AI MODEL ===
@st.cache_resource
def train_model(df_perf, df_gmv):
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    
    df_perf["month_num"] = df_perf["Month"].map(month_map)
    df_perf["date"] = pd.to_datetime(dict(year=df_perf["Year"], month=df_perf["month_num"], day=1))
    df_perf["year_month"] = df_perf["date"].dt.to_period("M").astype(str)

    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M").astype(str)

    def get_campaign_type(date):
        if pd.isna(date): return "unknown"
        if date.day == date.month: return "dday"
        elif date.day == 15: return "midmonth"
        elif date.day == 25: return "payday"
        else: return "normal_day"
    
    df_gmv["campaign_type"] = df_gmv["Data"].apply(get_campaign_type)

    campaign_map = df_gmv[["year_month", "campaign_type"]].drop_duplicates()
    df = pd.merge(df_perf, campaign_map, on="year_month", how="left")

    df = df.rename(columns={
        "Product": "product_name", "Brand": "brand", "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb",
        "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })

    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")

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
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5)
    model.fit(X, y)
    return model, summary, le

# === FORECAST ===
def forecast_future(model, base_df, le, months_ahead=3):
    today = datetime.today().replace(day=1)
    future_dates = [(today + relativedelta(months=i)).strftime("%Y-%m") for i in range(1, months_ahead + 1)]

    full_data = []
    for m in future_dates:
        temp = base_df.copy()
        temp["year_month"] = m
        temp["campaign_type"] = "dday" if int(m[-2:]) == 1 else "midmonth"
        for col in ["brand", "product_name", "platform", "campaign_type", "year_month"]:
            temp[col + "_enc"] = le.fit_transform(temp[col])
        full_data.append(temp)
    future_df = pd.concat(full_data)
    X_pred = future_df[[c for c in future_df.columns if "_enc" in c]]
    future_df["forecast_sales"] = model.predict(X_pred)
    return future_df

# === AI Recommendation ===
def recommend_insights(df):
    top = df.sort_values("forecast_sales", ascending=False).iloc[0]
    msg = f"💡 **แนะนำให้โปรโมต SKU:** `{top['product_name']}` บน `{top['platform']}`\n"
    msg += f"เพราะเคยทำยอดขายได้ดีในช่วง `{top['campaign_type']}` และมียอดเฉลี่ยสูงกว่าปกติ"
    return msg

# === UI ===
st.title("📈 Shopee + Lazada AI Sales Forecast")

uploaded = st.sidebar.file_uploader("📂 Upload Excel File", type="xlsx")
months = st.sidebar.slider("⏱️ ทำนายล่วงหน้า (เดือน)", 1, 60, 6)

if uploaded:
    data = load_excel(uploaded)
    perf_df = next((data[k] for k in data if "Perf" in k or "Performance" in k), None)
    gmv_df = next((data[k] for k in data if "GMV" in k), None)

    if perf_df is not None and gmv_df is not None:
        model, summary, le = train_model(perf_df, gmv_df)
        forecast_df = forecast_future(model, summary, le, months_ahead=months)

        st.subheader("📊 Summary Dashboard")
        col1, col2, col3 = st.columns(3)
        col1.metric("🔮 ยอดขายรวมล่วงหน้า", f"{forecast_df['forecast_sales'].sum():,.0f} THB")
        col2.metric("📦 Total SKUs", forecast_df["product_name"].nunique())
        col3.metric("🕐 ช่วงเวลาทำนาย", f"{months} เดือน")

        st.markdown("---")
        st.subheader("📈 แนวโน้มยอดขายล่วงหน้า (รายเดือน)")
        trend = forecast_df.groupby(["year_month", "platform"])["forecast_sales"].sum().reset_index()
        fig = px.line(trend, x="year_month", y="forecast_sales", color="platform",
                      markers=True, title="แนวโน้มยอดขายล่วงหน้า")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏆 สินค้าขายดีล่วงหน้า")
        top_df = forecast_df.sort_values("forecast_sales", ascending=False).head(10)
        st.dataframe(top_df[["product_name", "platform", "forecast_sales", "campaign_type"]],
                     use_container_width=True)

        st.subheader("💡 AI แนะนำช่วงโปรโมต")
        st.markdown(recommend_insights(forecast_df))
    else:
        st.error("⚠️ ไม่พบข้อมูลที่ถูกต้อง กรุณาใช้ไฟล์ที่มีชีท Performance และ GMV")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import calendar

st.set_page_config(page_title="📊 AI Forecasting Dashboard", layout="wide")

def classify_campaign_type(date):
    if pd.isna(date): return "unknown"
    if date.day == date.month: return "dday"
    elif date.day == 15: return "midmonth"
    elif date.day == 25: return "payday"
    else: return "normal_day"

@st.cache_data
def load_excel(file):
    xls = pd.ExcelFile(file)
    data = {}
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()
            data[sheet] = df
        except: continue
    return data

def preprocess(df_perf, df_gmv):
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
    df_gmv["campaign_type"] = df_gmv["Data"].apply(classify_campaign_type)
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M")

    month_map = {v: k for k,v in enumerate(calendar.month_abbr) if v}
    df_perf["month_num"] = df_perf["Month"].map(lambda x: month_map.get(x[:3].upper(), 1))
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
    return df

def train_model(df):
    df = df.dropna(subset=["sales_thb"])
    le = LabelEncoder()
    df["brand_enc"] = le.fit_transform(df["brand"].astype(str))
    df["product_enc"] = le.fit_transform(df["product_name"].astype(str))
    df["platform_enc"] = le.fit_transform(df["platform"].astype(str))
    df["campaign_enc"] = le.fit_transform(df["campaign_type"].astype(str))
    df["month_enc"] = le.fit_transform(df["year_month"].astype(str))

    X = df[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc"]]
    y = df["sales_thb"]

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5)
    model.fit(X, y)
    return model, le

def forecast_future(df, model, le, months_ahead=6):
    latest = df["year_month"].sort_values().unique()[-1]
    year, month = map(int, latest.split("-"))
    future = []

    for i in range(1, months_ahead + 1):
        m = month + i
        y = year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        ym = f"{y}-{m:02d}"
        temp = df.copy()
        temp["year_month"] = ym
        future.append(temp)

    df_future = pd.concat(future, ignore_index=True)

    # 🔧 ทำ Label Encoding อีกครั้ง
    le = LabelEncoder()
    df_future["brand_enc"] = le.fit_transform(df_future["brand"].astype(str))
    df_future["product_enc"] = le.fit_transform(df_future["product_name"].astype(str))
    df_future["platform_enc"] = le.fit_transform(df_future["platform"].astype(str))
    df_future["campaign_enc"] = le.fit_transform(df_future["campaign_type"].astype(str))
    df_future["month_enc"] = le.fit_transform(df_future["year_month"].astype(str))

    # 🔮 ทำนาย
    X_future = df_future[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc"]]
    df_future["forecast_sales"] = model.predict(X_future)
    return df_future

def ai_recommendation(df):
    top = df.groupby("campaign_type")["forecast_sales"].sum().idxmax()
    reason = f"จากข้อมูล AI พบว่าแคมเปญ `{top}` มียอดขายสูงสุดในช่วงที่เลือก ✅"
    return reason

# === UI ===
st.title("🧠 AI Sales & Product Forecasting Dashboard")

uploaded_file = st.sidebar.file_uploader("📂 Upload .xlsx file", type=["xlsx"])
months_to_predict = st.sidebar.slider("🔮 ทำนายล่วงหน้า (เดือน)", 1, 60, 6)

if uploaded_file:
    sheets = load_excel(uploaded_file)
    perf = [k for k in sheets if "Perf" in k or "Performance" in k]
    gmv = [k for k in sheets if "GMV" in k]
    
    if perf and gmv:
        df_raw = preprocess(sheets[perf[0]], sheets[gmv[0]])
        model, le = train_model(df_raw)
        df_forecast = forecast_future(df_raw, model, le, months_ahead=months_to_predict)

        selected_month = st.sidebar.selectbox("เลือกเดือน", sorted(df_forecast["year_month"].unique()))
        selected_platform = st.sidebar.selectbox("เลือกแพลตฟอร์ม", ["All"] + sorted(df_forecast["platform"].unique()))

        df_filtered = df_forecast[df_forecast["year_month"] == selected_month]
        if selected_platform != "All":
            df_filtered = df_filtered[df_filtered["platform"] == selected_platform]

        # === Dashboard Cards ===
        total = df_filtered["forecast_sales"].sum()
        avg = df_filtered["forecast_sales"].mean()
        count = len(df_filtered)

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 ยอดขายรวม", f"{total:,.2f} THB")
        col2.metric("📦 SKU ทั้งหมด", f"{count:,}")
        col3.metric("📈 ยอดเฉลี่ย / SKU", f"{avg:,.2f} THB")

        # === Graphs ===
        st.subheader("📊 กราฟแนวโน้มยอดขาย")
        trend = df_forecast.groupby(["year_month", "platform"])["forecast_sales"].sum().reset_index()
        fig_trend = px.line(trend, x="year_month", y="forecast_sales", color="platform", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("🏆 สินค้าขายดีที่คาดการณ์ไว้")
        top = df_filtered.sort_values("forecast_sales", ascending=False).head(30)
        fig_bar = px.bar(top, x="forecast_sales", y="product_name", orientation="h", color="platform")
        st.plotly_chart(fig_bar, use_container_width=True)

        # === Recommendation ===
        st.markdown("💡 **AI แนะนำช่วงเวลาโปรโมต**")
        st.info(ai_recommendation(df_filtered))

        st.subheader("📄 รายละเอียดทั้งหมด")
        st.dataframe(df_filtered, use_container_width=True)

    else:
        st.error("❌ ไม่พบชีทข้อมูล Perf/GMV กรุณาอัปโหลดไฟล์ที่ถูกต้อง")
else:
    st.warning("⏳ กรุณาอัปโหลดไฟล์ Excel ก่อนเริ่มต้น")

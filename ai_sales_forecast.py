import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose

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

    # Time Series Decomposition for Trend Feature
    def get_trend(series):
        try:
            decomposition = seasonal_decompose(series, model='additive', extrapolate_trend='freq')
            return decomposition.trend.iloc[-1]  # Get the latest trend value
        except:
            return 0  # Handle cases where decomposition fails

    trend_data = summary.groupby("year_month")["sales_thb"].sum().reset_index()
    trend_data["trend"] = trend_data["sales_thb"].rolling(window=3).apply(get_trend, raw=False) # Simple rolling trend
    trend_map = dict(zip(trend_data["year_month"], trend_data["trend"]))
    summary["trend"] = summary["year_month"].map(trend_map).fillna(0)  # Fill NaN with 0

    X = summary[[c for c in summary.columns if "_enc" in c] + ["trend"]]  # Include trend
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

        # Add Trend Feature to Forecast
        trend_value = base_df[base_df["year_month"] == base_df["year_month"].max()]["trend"].iloc[0] # Use last known trend
        temp["trend"] = trend_value # Assume trend continues (can be improved)
        full_data.append(temp)
    future_df = pd.concat(full_data)
    X_pred = future_df[[c for c in future_df.columns if "_enc" in c] + ["trend"]] # Include trend
    future_df["forecast_sales"] = model.predict(X_pred)
    return future_df

# === AI Recommendation ===
def recommend_insights(df):
    top = df.sort_values("forecast_sales", ascending=False).iloc[0]
    msg = f"💡 **แนะนำให้โปรโมต SKU:** `{top['product_name']}` บน `{top['platform']}`\n"
    
    # Add Reasons for Recommendation
    reasons = []
    
    # Campaign Performance
    campaign_history = df[(df["product_name"] == top["product_name"]) & (df["platform"] == top["platform"]) & (df["campaign_type"] == top["campaign_type"])]
    if not campaign_history.empty:
        avg_sales = campaign_history["sales_thb"].mean()
        if avg_sales > df["sales_thb"].mean():
            reasons.append(f"เคยทำยอดขายได้ดีในช่วง `{top['campaign_type']}` และมียอดเฉลี่ยสูงกว่าปกติ")
    
    # Sales Trend
    trend_history = df[(df["product_name"] == top["product_name"]) & (df["platform"] == top["platform"])].sort_values("year_month")
    if len(trend_history) >= 3:
        sales_trend = trend_history["sales_thb"].diff().dropna()
        if (sales_trend[-1] > 0).bool() and (sales_trend[-2] > 0).bool():
            reasons.append("ยอดขายเติบโตต่อเนื่องในช่วง 2 เดือนที่ผ่านมา")
    
    # Platform Comparison
    platform_comparison = df[df["product_name"] == top["product_name"]].groupby("platform")["sales_thb"].mean()
    other_platform = platform_comparison.drop(top["platform"], errors='ignore')
    if not other_platform.empty:
        if top["sales_thb"] > other_platform.max():
            platform_name = other_platform.idxmax()
            percent_diff = (top["sales_thb"] - other_platform.max()) / other_platform.max() * 100
            reasons.append(f"SKU นี้ขายดีกว่าบน {top['platform']} เมื่อเทียบกับ {platform_name} (+{percent_diff:.0f}%)")
    
    if reasons:
        msg += " เพราะ: " + ", ".join(reasons)
    else:
        msg += " เพราะมีศักยภาพในการทำยอดขายได้ดี"
    
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
        col1, col2, col3, col4 = st.columns(4)

        # Calculate Summary Metrics
        total_forecast = forecast_df["forecast_sales"].sum()
        avg_forecast = forecast_df["forecast_sales"].mean()
        
        # Platform Comparison
        platform_sales = forecast_df.groupby("platform")["forecast_sales"].sum()
        shopee_sales = platform_sales.get("Shopee", 0)
        lazada_sales = platform_sales.get("Lazada", 0)
        platform_diff_percent = 0
        if max(shopee_sales, lazada_sales) > 0:
            platform_diff_percent = abs(shopee_sales - lazada_sales) / max(shopee_sales, lazada_sales) * 100
        
        col1.metric("🔮 ยอดขายรวมล่วงหน้า", f"{total_forecast:,.0f} THB")
        col2.metric("📦 Total SKUs", forecast_df["product_name"].nunique())
        col3.metric("🕐 ช่วงเวลาทำนาย", f"{months} เดือน")
        col4.metric("💰 ยอดขายเฉลี่ย", f"{avg_forecast:,.0f} THB")

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

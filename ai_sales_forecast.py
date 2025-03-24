import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose

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
    # --- Clean GMV ---
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
    df_gmv["campaign_type"] = df_gmv["Data"].apply(lambda d: "unknown" if pd.isna(d) else (
        "dday" if d.day == d.month else "midmonth" if d.day == 15 else "payday" if d.day == 25 else "normal_day"))
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M")

    # --- Clean Performance ---
    month_map = {m.upper(): i for i, m in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], start=1)}
    df_perf["month_num"] = df_perf["Month"].str[:3].str.upper().map(month_map)
    df_perf["date"] = pd.to_datetime(dict(year=df_perf["Year"], month=df_perf["month_num"], day=1), errors="coerce")
    df_perf["year_month"] = df_perf["date"].dt.to_period("M")

    # --- Merge ---
    df = pd.merge(df_perf, df_gmv[["year_month", "campaign_type"]].drop_duplicates(), on="year_month", how="left")
    df = df.rename(columns={
        "Product": "product_name", "Brand": "brand", "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb", "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["year_month"].astype(str)
    df = df.dropna(subset=["sales_thb", "brand", "product_name", "platform", "campaign_type"])

    # --- Growth Rate ---
    df["month_numeric"] = df["year_month"].apply(lambda x: int(x.replace("-", "")))
    growth_rates = df.groupby(["product_name", "platform"]).apply(
        lambda g: g.sort_values("month_numeric").assign(
            pct_change=g["sales_thb"].pct_change().fillna(0)
        )["pct_change"].mean()
    ).reset_index(name="avg_growth_rate")

    # --- Summary ---
    summary = df.groupby(["brand", "product_name", "platform", "year_month", "campaign_type"]).agg({
        "sales_thb": "sum", "units_sold": "sum", "conversion_rate": "mean"
    }).reset_index()

    summary = pd.merge(summary, growth_rates, on=["product_name", "platform"], how="left").fillna(0)

    # ✅ NEW: Calculate Trend using seasonal_decompose
    trend_list = []
    for (product, platform), group in summary.groupby(["product_name", "platform"]):
        ts = group.sort_values("year_month").set_index("year_month")["sales_thb"]
        ts.index = pd.PeriodIndex(ts.index, freq="M")
        if len(ts) >= 6:  # ต้องมีข้อมูลอย่างน้อย 6 จุดเพื่อให้ seasonal_decompose ทำงานได้
            try:
                result = seasonal_decompose(ts, model='additive', period=3, extrapolate_trend='freq')
                trend_values = result.trend.fillna(method='bfill').fillna(method='ffill')  # เติม NaN
            except:
                trend_values = ts.copy()
        else:
            trend_values = ts.copy()

        trend_df = trend_values.reset_index()
        trend_df["product_name"] = product
        trend_df["platform"] = platform
        trend_df = trend_df.rename(columns={"sales_thb": "trend"})
        trend_list.append(trend_df)

    trend_all = pd.concat(trend_list)
    trend_all["year_month"] = trend_all["year_month"].astype(str)

    # ✅ Merge trend into summary
    summary = pd.merge(summary, trend_all, on=["year_month", "product_name", "platform"], how="left")
    summary["trend"] = summary["trend"].fillna(method='bfill').fillna(method='ffill')

    # --- Encode ---
    le_brand, le_product, le_platform, le_campaign = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
    summary["month_enc"] = summary["year_month"].apply(lambda x: int(x.replace("-", "")))
    summary["brand_enc"] = le_brand.fit_transform(summary["brand"])
    summary["product_enc"] = le_product.fit_transform(summary["product_name"])
    summary["platform_enc"] = le_platform.fit_transform(summary["platform"])
    summary["campaign_enc"] = le_campaign.fit_transform(summary["campaign_type"])

    # ✅ Train Model
    features = ["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate", "trend"]
    X = summary[features]
    y = summary["sales_thb"]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=6)
    model.fit(X, y)

    encoders = {"brand": le_brand, "product": le_product, "platform": le_platform, "campaign": le_campaign}
    return model, summary, encoders

def future_data(df, months_ahead, le):
    future_months = pd.date_range(datetime.today(), periods=months_ahead, freq='MS').to_period("M").astype(str)
    future_rows = []
    unique = df[["brand", "product_name", "platform", "campaign_type"]].drop_duplicates()
    for month in future_months:
        for _, row in unique.iterrows():
            future_rows.append({
                "brand": row["brand"],
                "product_name": row["product_name"],
                "platform": row["platform"],
                "campaign_type": row["campaign_type"],
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
        df_trend = df_future.groupby("year_month")["forecast_sales"].sum().reset_index()
        fig1 = px.line(df_trend, x="year_month", y="forecast_sales", markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🏆 Top Forecasted Products")
        top_products = df_future.groupby("product_name")["forecast_sales"].sum().sort_values(ascending=False).head(20).reset_index()
        fig2 = px.bar(top_products, x="forecast_sales", y="product_name", orientation="h", height=600)
        st.plotly_chart(fig2, use_container_width=True)

        # === Platform Comparison ===
        st.subheader("📊 Forecasted Sales by Platform")
        platform_summary = df_future.groupby("platform")["forecast_sales"].sum().reset_index()
        fig3 = px.pie(platform_summary, names="platform", values="forecast_sales", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)

        # === AI Recommendation ===
        st.subheader("💡 AI แนะนำช่วงเวลาที่ควรโปรโมต")
        recommend = df_future.groupby(["year_month", "campaign_type"])["forecast_sales"].sum().reset_index()
        top_recommend = recommend.sort_values("forecast_sales", ascending=False).head(3)
        for _, row in top_recommend.iterrows():
            st.info(f"📅 เดือน `{row['year_month']}` แคมเปญ `{row['campaign_type']}` คาดว่าจะทำยอดขายได้สูงถึง **{row['forecast_sales']:,.0f} THB** 💥")

    else:
        st.error("❗ ไม่พบ Sheet 'Performance' หรือ 'GMV'")
else:
    st.info("📤 กรุณาอัปโหลดไฟล์ Excel (.xlsx)")

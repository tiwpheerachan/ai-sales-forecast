import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import calendar

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def train_model(df_perf, df_gmv):
    # Clean GMV
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
    df_gmv["campaign_type"] = df_gmv["Data"].apply(lambda d: "unknown" if pd.isna(d) else (
        "dday" if d.day == d.month else "midmonth" if d.day == 15 else "payday" if d.day == 25 else "normal_day"))
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M")

    # Clean Performance
    month_map = {m.upper(): i for i, m in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], start=1)}
    df_perf["month_num"] = df_perf["Month"].str[:3].str.upper().map(month_map)
    df_perf["date"] = pd.to_datetime(dict(year=df_perf["Year"], month=df_perf["month_num"], day=1), errors="coerce")
    df_perf["year_month"] = df_perf["date"].dt.to_period("M")

    # Merge + Clean
    df = pd.merge(df_perf, df_gmv[["year_month", "campaign_type"]].drop_duplicates(), on="year_month", how="left")
    df = df.rename(columns={
        "Product": "product_name", "Brand": "brand", "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb", "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["year_month"].astype(str)
    df = df.dropna(subset=["sales_thb", "brand", "product_name", "platform", "campaign_type"])

    # Calculate Growth Rate
    df["month_numeric"] = df["year_month"].apply(lambda x: int(x.replace("-", "")))
    growth_rates = df.groupby(["product_name", "platform"]).apply(
        lambda g: g.sort_values("month_numeric").assign(
            pct_change=g["sales_thb"].pct_change().fillna(0)
        )["pct_change"].mean()
    ).reset_index(name="avg_growth_rate")

    # Summary
    summary = df.groupby(["brand", "product_name", "platform", "year_month", "campaign_type"]).agg({
        "sales_thb": "sum", "units_sold": "sum", "conversion_rate": "mean"
    }).reset_index()
    summary = pd.merge(summary, growth_rates, on=["product_name", "platform"], how="left").fillna(0)

    # Encode
    le_brand, le_product, le_platform, le_campaign = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
    summary["month_enc"] = summary["year_month"].apply(lambda x: int(x.replace("-", "")))
    summary["brand_enc"] = le_brand.fit_transform(summary["brand"])
    summary["product_enc"] = le_product.fit_transform(summary["product_name"])
    summary["platform_enc"] = le_platform.fit_transform(summary["platform"])
    summary["campaign_enc"] = le_campaign.fit_transform(summary["campaign_type"])

    # Train Model
    X = summary[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate"]]
    y = summary["sales_thb"]
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=6)
    model.fit(X, y)

    encoders = {"brand": le_brand, "product": le_product, "platform": le_platform, "campaign": le_campaign}
    return model, summary, encoders

def forecast_future(summary, model, encoders, months_ahead, growth_expectation=1.0):
    from datetime import datetime
    future_dates = pd.date_range(datetime.today(), periods=months_ahead, freq="MS").to_period("M").astype(str)
    base = summary[["brand", "product_name", "platform", "campaign_type"]].drop_duplicates()

    rows = []
    for month in future_dates:
        for _, row in base.iterrows():
            rows.append({
                "brand": row["brand"], "product_name": row["product_name"],
                "platform": row["platform"], "campaign_type": row["campaign_type"],
                "year_month": month
            })

    future = pd.DataFrame(rows)
    future["brand_enc"] = encoders["brand"].transform(future["brand"])
    future["product_enc"] = encoders["product"].transform(future["product_name"])
    future["platform_enc"] = encoders["platform"].transform(future["platform"])
    future["campaign_enc"] = encoders["campaign"].transform(future["campaign_type"])
    future["month_enc"] = future["year_month"].apply(lambda x: int(x.replace("-", "")))

    # เติม growth rate จาก summary
    growth_lookup = summary.groupby(["product_name", "platform"])["avg_growth_rate"].mean().reset_index()
    future = pd.merge(future, growth_lookup, on=["product_name", "platform"], how="left").fillna(0)
    future["adjusted_growth"] = future["avg_growth_rate"] * growth_expectation

    X = future[["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "adjusted_growth"]]
    future["forecast_sales"] = model.predict(X)
    return future

# === Streamlit UI ===
st.title("🧠 AI Sales & Product Forecasting Dashboard")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
months_to_predict = st.sidebar.slider("🔮 Forecast Months Ahead", 1, 60, 3)

if uploaded_file:
    dfs = load_excel(uploaded_file)
    perf_sheet = next((k for k in dfs if "Perf" in k or "Performance" in k), None)
    gmv_sheet = next((k for k in dfs if "GMV" in k), None)

    if perf_sheet and gmv_sheet:
        df_perf = dfs[perf_sheet]
        df_gmv = dfs[gmv_sheet]
        model, summary, encoders = train_model(df_perf, df_gmv)
        df_future = forecast_future(summary, model, encoders, months_to_predict)

        platform = st.sidebar.selectbox("เลือก Platform", ["All"] + sorted(df_future["platform"].unique()))
        if platform != "All":
            df_future = df_future[df_future["platform"] == platform]

        # Metrics
        total_sales = df_future["forecast_sales"].sum()
        total_skus = df_future["product_name"].nunique()
        avg_sales = total_sales / total_skus if total_skus > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Forecasted Sales", f"{total_sales:,.0f} THB")
        col2.metric("📦 Total SKUs", f"{total_skus}")
        col3.metric("📈 Avg. Sales/SKU", f"{avg_sales:,.2f} THB")

        # Graph 1: Forecast Trend
        st.subheader("📈 Forecasted Sales Trend by Month")
        df_trend = df_future.groupby("year_month")["forecast_sales"].sum().reset_index()
        fig1 = px.line(df_trend, x="year_month", y="forecast_sales", markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        # Graph 2: Top Products
        st.subheader("🏆 Top Forecasted Products")
        top_products = df_future.groupby("product_name")["forecast_sales"].sum().sort_values(ascending=False).head(15).reset_index()
        fig2 = px.bar(top_products, x="forecast_sales", y="product_name", orientation="h", title="Top Products")
        st.plotly_chart(fig2, use_container_width=True)

        # Graph 3: Platform Comparison
        st.subheader("📊 Forecasted Sales by Platform")
        platform_summary = df_future.groupby("platform")["forecast_sales"].sum().reset_index()
        fig3 = px.pie(platform_summary, names="platform", values="forecast_sales", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)

        # AI Recommendation
        st.subheader("💡 AI แนะนำช่วงเวลาที่ควรโปรโมต")
        rec = df_future.groupby(["year_month", "campaign_type"])["forecast_sales"].sum().reset_index()
        top_rec = rec.sort_values("forecast_sales", ascending=False).head(3)
        for _, row in top_rec.iterrows():
            st.info(f"✨ เดือน `{row['year_month']}` แคมเปญ `{row['campaign_type']}` มีแนวโน้มยอดขายสูงถึง **{row['forecast_sales']:,.0f} THB**")

        # Table
        st.subheader("📄 Forecast Table (All SKUs)")
        st.dataframe(df_future, use_container_width=True)

    else:
        st.warning("⚠️ ไม่พบ Sheet ที่ชื่อ Performance หรือ GMV")
else:
    st.info("📤 กรุณาอัปโหลดไฟล์ Excel เพื่อเริ่มต้น")

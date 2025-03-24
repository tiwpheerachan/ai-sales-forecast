import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="📊 AI Sales & Product Forecasting", layout="wide")

@st.cache_resource
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
    # === CLEAN GMV ===
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
    df_gmv["campaign_type"] = df_gmv["Data"].apply(lambda d: "unknown" if pd.isna(d) else (
        "dday" if d.day == d.month else "midmonth" if d.day == 15 else "payday" if d.day == 25 else "normal_day"))
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M")

    # === CLEAN PERFORMANCE ===
    month_map = {m.upper(): i for i, m in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], start=1)}
    df_perf["month_num"] = df_perf["Month"].str[:3].str.upper().map(month_map)
    df_perf["date"] = pd.to_datetime(dict(year=df_perf["Year"], month=df_perf["month_num"], day=1), errors="coerce")
    df_perf["year_month"] = df_perf["date"].dt.to_period("M")

    # === MERGE ===
    df = pd.merge(df_perf, df_gmv[["year_month", "campaign_type"]].drop_duplicates(), on="year_month", how="left")
    df = df.rename(columns={
        "Product": "product_name", "Brand": "brand", "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb", "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["year_month"].astype(str)
    df = df.dropna(subset=["sales_thb", "brand", "product_name", "platform", "campaign_type"])

    # === GROWTH RATE ===
    df["month_numeric"] = df["year_month"].apply(lambda x: int(x.replace("-", "")))
    growth_rates = df.groupby(["product_name", "platform"]).apply(
        lambda g: g.sort_values("month_numeric").assign(
            pct_change=g["sales_thb"].pct_change().fillna(0)
        )["pct_change"].mean()
    ).reset_index(name="avg_growth_rate")

    # === SUMMARY ===
    summary = df.groupby(["brand", "product_name", "platform", "year_month", "campaign_type"]).agg({
        "sales_thb": "sum", "units_sold": "sum", "conversion_rate": "mean"
    }).reset_index()
    summary = pd.merge(summary, growth_rates, on=["product_name", "platform"], how="left").fillna(0)

    # === TREND (seasonal_decompose) ===
    trend_list = []
    for (product, platform), group in summary.groupby(["product_name", "platform"]):
        ts = group.sort_values("year_month").set_index("year_month")["sales_thb"]
        ts.index = pd.PeriodIndex(ts.index, freq="M")
        if len(ts) >= 6:
            try:
                result = seasonal_decompose(ts, model='additive', period=3, extrapolate_trend='freq')
                trend = result.trend.fillna(method='bfill').fillna(method='ffill')
            except:
                trend = ts.copy()
        else:
            trend = ts.copy()

        trend_df = trend.reset_index()
        trend_df["product_name"] = product
        trend_df["platform"] = platform
        trend_df = trend_df.rename(columns={"sales_thb": "trend"})
        trend_list.append(trend_df)

    trend_all = pd.concat(trend_list)
    trend_all["year_month"] = trend_all["year_month"].astype(str)
    summary = pd.merge(summary, trend_all, on=["year_month", "product_name", "platform"], how="left")
    summary["trend"] = summary["trend"].fillna(method='bfill').fillna(method='ffill')

    # === ENCODING ===
    le_brand = LabelEncoder()
    le_product = LabelEncoder()
    le_platform = LabelEncoder()
    le_campaign = LabelEncoder()
    summary["month_enc"] = summary["year_month"].apply(lambda x: int(x.replace("-", "")))
    summary["brand_enc"] = le_brand.fit_transform(summary["brand"])
    summary["product_enc"] = le_product.fit_transform(summary["product_name"])
    summary["platform_enc"] = le_platform.fit_transform(summary["platform"])
    summary["campaign_enc"] = le_campaign.fit_transform(summary["campaign_type"])

# === TRAIN MODEL ===
    features = ["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate", "trend"]
    X = summary[features].replace([np.inf, -np.inf], np.nan).dropna()
    y = summary.loc[X.index, "sales_thb"]

    # Hyperparameter Tuning with GridSearchCV and Cross-Validation
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = GradientBoostingRegressor()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best model
    best_model = grid_search.best_estimator_

    encoders = {
        "brand": le_brand, "product": le_product,
        "platform": le_platform, "campaign": le_campaign
    }
    return best_model, summary, encoders

def forecast_future(summary, model, encoders, months_ahead):
    future_months = pd.date_range(datetime.today(), periods=months_ahead, freq="MS").to_period("M").astype(str)
    base = summary[["brand", "product_name", "platform", "campaign_type"]].drop_duplicates()

    rows = []
    for month in future_months:
        for _, row in base.iterrows():
            rows.append({
                "brand": row["brand"],
                "product_name": row["product_name"],
                "platform": row["platform"],
                "campaign_type": row["campaign_type"],
                "year_month": month
            })

    future = pd.DataFrame(rows)

    # === ENCODE FUTURE ===
    future["brand_enc"] = encoders["brand"].transform(future["brand"])
    future["product_enc"] = encoders["product"].transform(future["product_name"])
    future["platform_enc"] = encoders["platform"].transform(future["platform"])
    future["campaign_enc"] = encoders["campaign"].transform(future["campaign_type"])
    future["month_enc"] = future["year_month"].apply(lambda x: int(x.replace("-", "")))

    # === ADD avg_growth_rate & trend ===
    growth_lookup = summary.groupby(["product_name", "platform"])["avg_growth_rate"].mean().reset_index()
    trend_lookup = summary.groupby(["product_name", "platform"])["trend"].mean().reset_index()
    future = pd.merge(future, growth_lookup, on=["product_name", "platform"], how="left")
    future = pd.merge(future, trend_lookup, on=["product_name", "platform"], how="left")
    future["avg_growth_rate"] = future["avg_growth_rate"].fillna(0)
    future["trend"] = future["trend"].fillna(method='ffill').fillna(method='bfill')

    # === PREDICT ===
    features = ["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate", "trend"]
    X = future[features].replace([np.inf, -np.inf], np.nan).dropna()
    future = future.loc[X.index]
    future["forecast_sales"] = model.predict(X)

    return future

# === STREAMLIT UI ===
st.title("🧠 AI Sales & Product Forecasting Dashboard")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
months = st.sidebar.slider("🔮 Forecast Months Ahead", 1, 60, 3)

if uploaded_file:
    dfs = load_excel(uploaded_file)
    perf_sheet = next((s for s in dfs if "perf" in s.lower()), None)
    gmv_sheet = next((s for s in dfs if "gmv" in s.lower()), None)

    if perf_sheet and gmv_sheet:
        model, summary, encoders = train_model(dfs[perf_sheet], dfs[gmv_sheet])
        df_future = forecast_future(summary, model, encoders, months)

        st.sidebar.markdown("## 🎯 Filters")
        selected_platform = st.sidebar.selectbox("เลือกแพลตฟอร์ม", ["All"] + sorted(df_future["platform"].unique()))
        if selected_platform != "All":
            df_future = df_future[df_future["platform"] == selected_platform]

        # === METRICS ===
        total_sales = df_future["forecast_sales"].sum()
        total_skus = df_future["product_name"].nunique()
        avg_sales = total_sales / total_skus if total_skus else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Forecasted Sales", f"{total_sales:,.0f} THB")
        col2.metric("📦 Total SKUs", f"{total_skus}")
        col3.metric("📈 Avg. Sales/SKU", f"{avg_sales:,.2f} THB")

        # === TREND CHART ===
        st.subheader("📈 Forecasted Sales Trend")
        trend_df = df_future.groupby("year_month")["forecast_sales"].sum().reset_index()
        fig1 = px.line(trend_df, x="year_month", y="forecast_sales", markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        # === TOP PRODUCTS ===
        st.subheader("🏆 Top Products")
        top_products = df_future.groupby("product_name")["forecast_sales"].sum().sort_values(ascending=False).head(15).reset_index()
        fig2 = px.bar(top_products, x="forecast_sales", y="product_name", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

        # === PLATFORM PIE ===
        st.subheader("📊 Platform Share")
        platform_sum = df_future.groupby("platform")["forecast_sales"].sum().reset_index()
        fig3 = px.pie(platform_sum, names="platform", values="forecast_sales", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)

        # === RECOMMENDATION ===
        st.subheader("💡 AI แนะนำช่วงเวลาที่ควรโปรโมต")
        rec = df_future.groupby(["year_month", "campaign_type"])["forecast_sales"].sum().reset_index()
        top_rec = rec.sort_values("forecast_sales", ascending=False).head(3)
        for _, row in top_rec.iterrows():
            st.info(f"📅 เดือน `{row['year_month']}` แคมเปญ `{row['campaign_type']}` มียอดขายคาดการณ์สูงถึง **{row['forecast_sales']:,.0f} THB** 💥")

        st.subheader("📄 รายการทำนายทั้งหมด")
        st.dataframe(df_future, use_container_width=True)
    else:
        st.warning("❗ ไม่พบ Sheet ที่มีชื่อ 'Perf' หรือ 'GMV'")
else:
    st.info("📤 กรุณาอัปโหลดไฟล์ Excel (.xlsx) เพื่อเริ่มต้น")

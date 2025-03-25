
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import GridSearchCV, KFold
from openai import OpenAI
import openai
import pyttsx3

# ตั้งค่า OpenAI API Key จาก secrets
import openai
import pyttsx3

openai.api_key = st.secrets["openai_api_key"]

def recommend_insights(df_future, summary):
    st.subheader("💡 AI วิเคราะห์ & แนะนำช่วงเวลาที่ควรโปรโมต")

    top_campaigns = df_future.groupby(["year_month", "campaign_type"])["forecast_sales"].sum().reset_index()
    top_campaigns = top_campaigns.sort_values("forecast_sales", ascending=False).head(3)

    # แสดงแบบเบื้องต้น
    for _, row in top_campaigns.iterrows():
        st.success(
            f"📅 เดือน `{row['year_month']}` แคมเปญ `{row['campaign_type']}` "
            f"คาดการณ์ยอดขาย **{row['forecast_sales']:,.0f} THB**"
        )

    # สรุปด้วย GPT
    st.subheader("🧠 สรุปโดย LLM (GPT)")

    insight_prompt = (
        "คุณคือผู้ช่วยวิเคราะห์ยอดขายสินค้า AI โปรดสรุปว่าทำไมสินค้าด้านล่างถึงจะขายดี "
        "โดยอธิบายแนวโน้มยอดขายในแต่ละแคมเปญด้วยภาษาที่เข้าใจง่าย (ภาษาไทย):\n"
    )
    for _, row in top_campaigns.iterrows():
        insight_prompt += f"- เดือน {row['year_month']} แคมเปญ {row['campaign_type']} คาดว่าจะมียอดขาย {row['forecast_sales']:,.0f} บาท\n"

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": insight_prompt}],
            max_tokens=500,
            temperature=0.7
        )
        summary_text = response.choices[0].message.content.strip()
        st.info(summary_text)

        if st.button("🔊 อ่านออกเสียง"):
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # ปรับความเร็วการอ่าน
            engine.say(summary_text)
            engine.runAndWait()

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสรุปด้วย GPT: {e}")

@st.cache_data
def load_excel(file):
    xls = pd.ExcelFile(file)
    dfs = {}
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            df.columns = df.columns.str.strip()
            dfs[sheet] = df
        except Exception as e:
            print(f"❌ Error loading sheet {sheet}: {e}")
            continue
    return dfs

@st.cache_resource
def train_model(df_perf, df_gmv, fast_mode=False):
    df_gmv["Data"] = pd.to_datetime(df_gmv["Data"], errors="coerce")
    df_gmv["campaign_type"] = df_gmv["Data"].apply(lambda d: "unknown" if pd.isna(d) else (
        "dday" if d.day == d.month else "midmonth" if d.day == 15 else "payday" if d.day == 25 else "normal_day"))
    df_gmv["year_month"] = df_gmv["Data"].dt.to_period("M")

    month_map = {m.upper(): i for i, m in enumerate(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], start=1)}
    df_perf["month_num"] = df_perf["Month"].str[:3].str.upper().map(month_map)
    df_perf["date"] = pd.to_datetime(dict(year=df_perf["Year"], month=df_perf["month_num"], day=1), errors="coerce")
    df_perf["year_month"] = df_perf["date"].dt.to_period("M")

    df = pd.merge(df_perf, df_gmv[["year_month", "campaign_type"]].drop_duplicates(), on="year_month", how="left")
    df = df.rename(columns={
        "Product": "product_name", "Brand": "brand", "Platforms": "platform",
        "Sales (Confirmed Order) (THB)": "sales_thb", "Units (Confirmed Order)": "units_sold",
        "Conversion Rate (Confirmed Order)": "conversion_rate"
    })
    df["conversion_rate"] = pd.to_numeric(df["conversion_rate"], errors="coerce")
    df["year_month"] = df["year_month"].astype(str)
    df = df.dropna(subset=["sales_thb", "brand", "product_name", "platform", "campaign_type"])

    df["month_numeric"] = df["year_month"].apply(lambda x: int(x.replace("-", "")))
    growth_rates = df.groupby(["product_name", "platform"]).apply(
        lambda g: g.sort_values("month_numeric").assign(
            pct_change=g["sales_thb"].pct_change().fillna(0)
        )["pct_change"].mean()
    ).reset_index(name="avg_growth_rate")

    summary = df.groupby(["brand", "product_name", "platform", "year_month", "campaign_type"]).agg({
        "sales_thb": "sum", "units_sold": "sum", "conversion_rate": "mean"
    }).reset_index()
    summary = pd.merge(summary, growth_rates, on=["product_name", "platform"], how="left").fillna(0)

    le_brand = LabelEncoder()
    le_product = LabelEncoder()
    le_platform = LabelEncoder()
    le_campaign = LabelEncoder()
    summary["month_enc"] = summary["year_month"].apply(lambda x: int(x.replace("-", "")))
    summary["brand_enc"] = le_brand.fit_transform(summary["brand"])
    summary["product_enc"] = le_product.fit_transform(summary["product_name"])
    summary["platform_enc"] = le_platform.fit_transform(summary["platform"])
    summary["campaign_enc"] = le_campaign.fit_transform(summary["campaign_type"])

    features = ["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate"]
    X = summary[features].replace([np.inf, -np.inf], np.nan).dropna()
    y = summary.loc[X.index, "sales_thb"]

    if fast_mode:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
        model.fit(X, y)
    else:
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 4]
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=kf, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        model = grid_search.best_estimator_

    encoders = {"brand": le_brand, "product": le_product, "platform": le_platform, "campaign": le_campaign}
    return model, summary, encoders

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
    future["brand_enc"] = encoders["brand"].transform(future["brand"])
    future["product_enc"] = encoders["product"].transform(future["product_name"])
    future["platform_enc"] = encoders["platform"].transform(future["platform"])
    future["campaign_enc"] = encoders["campaign"].transform(future["campaign_type"])
    future["month_enc"] = future["year_month"].apply(lambda x: int(x.replace("-", "")))

    growth_lookup = summary.groupby(["product_name", "platform"])["avg_growth_rate"].mean().reset_index()
    future = pd.merge(future, growth_lookup, on=["product_name", "platform"], how="left")
    future["avg_growth_rate"] = future["avg_growth_rate"].fillna(0)

    features = ["brand_enc", "product_enc", "platform_enc", "campaign_enc", "month_enc", "avg_growth_rate"]
    X = future[features].replace([np.inf, -np.inf], np.nan).dropna()
    future = future.loc[X.index]
    future["forecast_sales"] = model.predict(X)

    return future

# UI
st.title("🧠 AI Sales & Product Forecasting + LLM + Voice")
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
months = st.sidebar.slider("🔮 Forecast Months Ahead", 1, 60, 3)
fast_mode = st.sidebar.checkbox("⚡ Fast Mode", value=False)

if uploaded_file:
    dfs = load_excel(uploaded_file)
    perf_sheet = next((s for s in dfs if "perf" in s.lower()), None)
    gmv_sheet = next((s for s in dfs if "gmv" in s.lower()), None)

    if perf_sheet and gmv_sheet:
        with st.spinner("🚀 AI กำลังวิเคราะห์ข้อมูล โปรดรอสักครู่..."):
            model, summary, encoders = train_model(dfs[perf_sheet], dfs[gmv_sheet], fast_mode)
            df_future = forecast_future(summary, model, encoders, months)

        platform = st.sidebar.selectbox("เลือกแพลตฟอร์ม", ["All"] + sorted(df_future["platform"].unique()))
        if platform != "All":
            df_future = df_future[df_future["platform"] == platform]

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Forecasted Sales", f"{df_future['forecast_sales'].sum():,.0f} THB")
        col2.metric("📦 Total SKUs", f"{df_future['product_name'].nunique()}")
        col3.metric("📈 Avg. Sales/SKU", f"{df_future['forecast_sales'].mean():,.2f} THB")

        st.subheader("📈 Forecast Trend")
        fig1 = px.line(df_future.groupby("year_month")["forecast_sales"].sum().reset_index(), x="year_month", y="forecast_sales", markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🏆 Top Products")
        top_products = df_future.groupby("product_name")["forecast_sales"].sum().sort_values(ascending=False).head(15).reset_index()
        fig2 = px.bar(top_products, x="forecast_sales", y="product_name", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📊 Platform Share")
        fig3 = px.pie(df_future.groupby("platform")["forecast_sales"].sum().reset_index(), names="platform", values="forecast_sales", hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)

        recommend_insights(df_future, summary)

        st.subheader("📄 Forecast Table")
        st.dataframe(df_future, use_container_width=True)
    else:
        st.warning("⚠️ ไม่พบ Sheet ที่ชื่อ 'Perf' หรือ 'GMV'")
else:
    st.info("📤 กรุณาอัปโหลดไฟล์ Excel เพื่อเริ่มต้น")

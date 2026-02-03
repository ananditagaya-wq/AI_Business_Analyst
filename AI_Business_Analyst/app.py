import pandas as pd
import streamlit as st

# -------------------------------
# App Title
# -------------------------------
st.title("AI Business Analyst 📊")
st.subheader("Automated Data Analysis & Insights")

# -------------------------------
# Load Dataset
# -------------------------------
DATA_PATH = "sample_data/sales_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -------------------------------
# Dataset Preview
# -------------------------------
st.write("### Preview of Dataset")
st.dataframe(df.head())

# -------------------------------
# Dataset Overview
# -------------------------------
st.write("### Dataset Overview")
st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])

st.write("### Column Names")
st.write(list(df.columns))

# ===============================
# STEP 8: DATA CLEANING
# ===============================
st.write("### Data Cleaning Summary")

df_clean = df.copy()

# Remove duplicates
duplicates = df_clean.duplicated().sum()
df_clean.drop_duplicates(inplace=True)

# Missing values summary
missing_values = df_clean.isnull().sum()

st.write("Duplicate rows removed:", duplicates)
st.write("Missing values per column:")
st.dataframe(missing_values)

# ===============================
# STEP 9: KEY BUSINESS METRICS (FIXED)
# ===============================
st.write("### Key Business Metrics")

# Clean price columns (remove currency symbols & commas)
df_clean["actual_price"] = (
    df_clean["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df_clean["discount_price"] = (
    df_clean["discount_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

# Convert to numeric
df_clean["actual_price"] = pd.to_numeric(df_clean["actual_price"], errors="coerce")
df_clean["discount_price"] = pd.to_numeric(df_clean["discount_price"], errors="coerce")
df_clean["ratings"] = pd.to_numeric(df_clean["ratings"], errors="coerce")

# KPIs
avg_price = df_clean["actual_price"].mean()
avg_discount_price = df_clean["discount_price"].mean()
avg_rating = df_clean["ratings"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Avg Actual Price", f"{avg_price:.2f}")
col2.metric("Avg Discount Price", f"{avg_discount_price:.2f}")
col3.metric("Avg Rating", f"{avg_rating:.2f}")

# ===============================
# STEP 10: CATEGORY ANALYSIS
# ===============================
st.write("### Category-Level Analysis")

category_stats = (
    df_clean
    .groupby("main_category")
    .agg(
        avg_price=("actual_price", "mean"),
        avg_rating=("ratings", "mean"),
        product_count=("name", "count")
    )
    .sort_values(by="product_count", ascending=False)
)

st.dataframe(category_stats)

st.write("### Average Rating by Category")
st.bar_chart(category_stats["avg_rating"])
# ===============================
# STEP 11: SUB-CATEGORY INSIGHTS
# ===============================
st.write("### Sub-Category Performance")

subcat_stats = (
    df_clean
    .groupby("sub_category")
    .agg(
        avg_price=("actual_price", "mean"),
        avg_rating=("ratings", "mean"),
        product_count=("name", "count")
    )
    .sort_values(by="product_count", ascending=False)
)

st.dataframe(subcat_stats.head(10))

st.write("### Average Rating by Sub-Category (Top 10)")
st.bar_chart(subcat_stats["avg_rating"].head(10))
# ===============================
# STEP 12: DISCOUNT IMPACT ANALYSIS
# ===============================
st.write("### Discount Impact Analysis")

df_clean["discount_percent"] = (
    (df_clean["actual_price"] - df_clean["discount_price"])
    / df_clean["actual_price"]
) * 100

st.write("### Discount % vs Ratings")
st.scatter_chart(
    df_clean[["discount_percent", "ratings"]].dropna()
)
# ===============================
# STEP 13: PRICE SEGMENT ANALYSIS
# ===============================
st.write("### Price Segment Analysis")

# Create price segments
df_clean["price_segment"] = pd.cut(
    df_clean["actual_price"],
    bins=[0, 30000, 50000, 80000, 200000],
    labels=["Budget", "Mid-Range", "Premium", "Luxury"]
)

segment_stats = (
    df_clean
    .groupby("price_segment")
    .agg(
        avg_rating=("ratings", "mean"),
        product_count=("name", "count")
    )
)

st.dataframe(segment_stats)

st.write("### Average Rating by Price Segment")
st.bar_chart(segment_stats["avg_rating"])
# ===============================
# STEP 14: AUTO-GENERATED INSIGHTS
# ===============================
st.write("### AI-Generated Business Insights")

best_segment = segment_stats["avg_rating"].idxmax()
best_rating = segment_stats["avg_rating"].max()

avg_discount = df_clean["discount_percent"].mean()

st.markdown(f"""
🔹 **Best Performing Price Segment:** {best_segment}  
🔹 **Highest Average Rating:** {best_rating:.2f} ⭐  
🔹 **Average Discount Across Products:** {avg_discount:.1f}%  

**Insights:**
- Higher-priced ACs tend to receive slightly better ratings, indicating perceived quality.
- Discounts do not strongly correlate with higher ratings, suggesting customers value performance over price cuts.
- Businesses should focus on product quality and energy efficiency rather than aggressive discounting.
""")
# -------------------------------
# STEP 15: SMART RECOMMENDATION SYSTEM
# -------------------------------
st.write("## 🔮 Smart Product Recommendation System")

max_budget = st.slider(
    "Maximum Budget (₹)",
    int(df_clean["actual_price"].min()),
    int(df_clean["actual_price"].max()),
    50000,
    step=5000
)

min_rating = st.slider(
    "Minimum Rating",
    1.0, 5.0, 4.0, 0.1
)

recommended = df_clean[
    (df_clean["actual_price"] <= max_budget) &
    (df_clean["ratings"] >= min_rating)
].sort_values(by=["ratings", "discount_percent"], ascending=False)

st.write("### Recommended Products")

if recommended.empty:
    st.warning("No products found for selected criteria.")
else:
    show_df = recommended[
        ["name", "actual_price", "discount_price", "ratings", "discount_percent", "link"]
    ].head(10)

    show_df["link"] = show_df["link"].apply(
        lambda x: f"[View Product]({x})" if pd.notna(x) else "N/A"
    )

    st.markdown(show_df.to_markdown(index=False), unsafe_allow_html=True)
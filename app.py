import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Lazy import transformers to avoid long startup when not needed
@st.cache_resource
def get_zero_shot_classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CATEGORIES = ["Food", "Utilities", "Travel", "Shopping", "Healthcare", "Others"]

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_sample_data(path="sample_transactions.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

def ensure_date_col(df):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        st.error("CSV must contain a 'Date' column.")
    return df

def categorize_transactions(df):
    classifier = get_zero_shot_classifier()
    # If Category column exists and non-empty, keep it
    if "Category" in df.columns and df["Category"].notna().any():
        return df
    descriptions = df["Description"].fillna("").astype(str).tolist()
    labels = []
    batch_size = 16
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        results = classifier(batch, CATEGORIES)
        # pipeline returns list of dicts for batch
        for r in results:
            labels.append(r["labels"][0])
    df["Category"] = labels
    return df

def summary_by_category(df, month_filter=None):
    # month_filter: tuple(year, month) or None
    dff = df.copy()
    if month_filter:
        year, month = month_filter
        dff = dff[(dff["Date"].dt.year == year) & (dff["Date"].dt.month == month)]
    summary = dff.groupby("Category")["Amount"].sum().reindex(CATEGORIES, fill_value=0)
    return summary

def month_bounds_from_latest(df):
    latest = df["Date"].max()
    if pd.isna(latest):
        return None, None
    # this month
    this = (latest.year, latest.month)
    # previous month - careful with January
    if latest.month == 1:
        prev = (latest.year - 1, 12)
    else:
        prev = (latest.year, latest.month - 1)
    return this, prev

def pct_change(prev, curr):
    if prev == 0 and curr == 0:
        return 0.0
    if prev == 0:
        return float("inf")
    return (curr - prev) / abs(prev) * 100.0

def generate_text_insights(category_summary, prev_summary, budgets):
    insights = []
    # top category
    top_cat = category_summary.idxmax()
    top_amt = category_summary.max()
    insights.append(f"Highest spend this month: **{top_cat}** — ₹{top_amt:,.2f}")

    # month-over-month comparison per category
    for cat in CATEGORIES:
        prev_amt = prev_summary.get(cat, 0.0)
        curr_amt = category_summary.get(cat, 0.0)
        change = pct_change(prev_amt, curr_amt)
        if change == float("inf"):
            insights.append(f"**{cat}**: New spending of ₹{curr_amt:,.2f} this month (no spend last month).")
        elif abs(change) >= 20:  # threshold for notable change
            sign = "increased" if change > 0 else "decreased"
            insights.append(f"**{cat}**: {sign} by {abs(change):.0f}% vs last month (₹{prev_amt:,.2f} → ₹{curr_amt:,.2f}).")

    # budget alerts
    for cat, limit in budgets.items():
        curr_amt = category_summary.get(cat, 0.0)
        if limit is not None and limit > 0 and curr_amt > limit:
            excess = curr_amt - limit
            insights.append(f"⚠️ **Budget alert**: {cat} over budget by ₹{excess:,.2f} (spent ₹{curr_amt:,.2f}, limit ₹{limit:,.2f}).")
    return insights

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Transaction & Spend Analyzer", layout="wide")
st.title("AI-Powered Transaction Categorizer & Spend Analyzer")
st.markdown("Upload your transactions CSV (Date, Description, Amount, Type). This app categorizes transactions, shows monthly comparisons, and notifies budget alerts.")

# Sidebar: data source & options
st.sidebar.header("Data & Settings")
use_sample = st.sidebar.checkbox("Use sample data", value=True)

uploaded_file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    except Exception:
        df = pd.read_csv(uploaded_file)
        df = ensure_date_col(df)
else:
    if use_sample:
        try:
            df = load_sample_data()
        except FileNotFoundError:
            st.error("Sample data file not found. Please upload your transactions CSV.")
            st.stop()
    else:
        st.info("Upload a CSV or enable sample data.")
        st.stop()

# Ensure columns are present
expected_cols = ["Date", "Description", "Amount", "Type"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"CSV missing expected columns: {missing}. Columns required: {expected_cols}")
    st.stop()

# Ensure numeric Amount
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
df = ensure_date_col(df)
# Show spinner during categorization
with st.spinner("Categorizing transactions... This may take a moment."):
    df = categorize_transactions(df)

# Preview
with st.expander("Preview transactions (first 10 rows)", expanded=False):
    st.dataframe(df.head(10))

# Compute month bounds
this_month, prev_month = month_bounds_from_latest(df)
if not this_month:
    st.error("No valid dates found in data.")
    st.stop()

# Show spinner while generating summaries
with st.spinner("Generating summaries and insights..."):
    this_summary = summary_by_category(df, month_filter=this_month)
    prev_summary = summary_by_category(df, month_filter=prev_month)

# Layout: Summary and charts
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Spending by Category (Current Month)")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    this_summary.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Amount (₹)")
    ax1.set_title(f"Spend by Category — {this_month[0]}-{this_month[1]:02d}")
    st.pyplot(fig1)

    # Top merchants
    st.subheader("Top merchants (current month)")
    curr_month_df = df[(df["Date"].dt.year == this_month[0]) & (df["Date"].dt.month == this_month[1])]
    top_merchants = curr_month_df.groupby("Description")["Amount"].sum().sort_values(ascending=False).head(5)
    st.table(top_merchants.reset_index().rename(columns={"Description":"Merchant","Amount":"Amount (₹)"}))

with col2:
    st.subheader("Month-over-Month Comparison")
    # Build DataFrame for plotting
    comp_df = pd.DataFrame({
        "Previous": prev_summary.reindex(CATEGORIES, fill_value=0).values,
        "Current": this_summary.reindex(CATEGORIES, fill_value=0).values
    }, index=CATEGORIES)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    comp_df.plot.bar(ax=ax2)
    ax2.set_ylabel("Amount (₹)")
    ax2.set_title(f"Previous vs Current Month — {prev_month[0]}-{prev_month[1]:02d} vs {this_month[0]}-{this_month[1]:02d}")
    st.pyplot(fig2)

    # Display percent change table
    pct_changes = {}
    for cat in CATEGORIES:
        prev_amt = prev_summary.get(cat, 0.0)
        curr_amt = this_summary.get(cat, 0.0)
        change = pct_change(prev_amt, curr_amt)
        if change == float("inf"):
            pct_str = "∞ (new)"
        else:
            pct_str = f"{change:.0f}%"
        pct_changes[cat] = {"Previous": prev_amt, "Current": curr_amt, "Change": pct_str}
    pct_table = pd.DataFrame.from_dict(pct_changes, orient="index")
    st.table(pct_table)

# Budget inputs
st.sidebar.header("Budget Settings (monthly)")
default_budgets = {c: 0 for c in CATEGORIES}
budget_limits = {}
st.sidebar.markdown("Set monthly budget per category (₹). Leave 0 for no budget.")
for c in CATEGORIES:
    val = st.sidebar.number_input(f"{c}", min_value=0.0, value=float(default_budgets[c]), step=100.0, format="%.2f")
    budget_limits[c] = val if val > 0 else None

# Generate insights and alerts
insights = generate_text_insights(this_summary, prev_summary, budget_limits)

# Show spinner during insights generation
with st.spinner("Analyzing and generating insights..."):
    insights = generate_text_insights(this_summary, prev_summary, budget_limits)

st.subheader("Generated Insights & Alerts")
for s in insights:
    if "⚠️" in s:
        st.error(s)
    else:
        st.write(s)

# Download categorized CSV
st.markdown("---")
st.subheader("Export")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download full categorized CSV", csv_bytes, file_name="categorized_transactions.csv", mime="text/csv")
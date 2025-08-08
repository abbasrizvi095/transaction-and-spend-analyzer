import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from categorize import categorize_transaction
from insights import generate_insights

st.title("AI-Powered Transaction Categorizer & Spend Analyzer")

uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Category" not in df.columns:
        st.write("Categorizing transactions...")
        df["Category"] = df["Description"].apply(categorize_transaction)

    st.subheader("Categorized Transactions")
    st.dataframe(df)

    st.subheader("Spending by Category")
    category_summary = df.groupby("Category")["Amount"].sum()
    fig, ax = plt.subplots()
    category_summary.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Insights")
    insights = generate_insights(df)
    for i in insights:
        st.write(f"- {i}")

    st.download_button("Download Categorized CSV", df.to_csv(index=False), "categorized_transactions.csv")

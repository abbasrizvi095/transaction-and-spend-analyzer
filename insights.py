import pandas as pd

def generate_insights(df):
    insights = []
    category_summary = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    top_category = category_summary.index[0]
    top_amount = category_summary.iloc[0]
    insights.append(f"Highest spend category: {top_category} (${top_amount:.2f})")

    if "Food" in category_summary.index and category_summary["Food"] > 500:
        insights.append("You spent over $500 on dining this month.")

    return insights

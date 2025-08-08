# 💳 Transaction & Spend Analyzer with AI Insights

An AI-powered Streamlit app that helps users track spending patterns, compare monthly trends, and receive budget alerts — designed as a **fintech product prototype** for portfolio demonstration.

---

## 🚀 Features
- 📊 **Category Spend Dashboard** — Interactive pie chart of spending distribution.
- 📅 **Monthly Trend Comparison** — See how this month’s spending stacks up against last month, per category.
- ⚠ **Budget Alerts** — Get notified when you exceed your budget in any category.
- 🧠 **AI Insights** — Natural language takeaways from your spending data.

---

## 🗂 Example Data
A sample dataset `sample_transactions.csv` is included with 100 transactions across categories:  
`Date, Description, Category, Amount`

---

## 🛠 Installation

```bash
# Clone the repo
git clone https://github.com/abbasrizvi095/transaction-and-spend-analyzer.git
cd transaction-and-spend-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

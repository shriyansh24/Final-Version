# 📊 CPI Analysis & Prediction Dashboard

Welcome to the **CPI (Cost Per Interview) Dashboard**, a Streamlit-powered web application that provides detailed analysis and predictive modeling of survey bid data. It helps users compare won vs. lost deals and recommend optimal CPI pricing strategies.

---

## 🚀 Features

- **Overview Dashboard**: Key metrics and interactive charts summarizing CPI, IR, and LOI differences.
- **CPI Analysis**: Explore CPI variation by Incidence Rate (IR), Length of Interview (LOI), and Sample Size.
- **CPI Prediction**: Machine learning models (Linear Regression, Random Forest, Gradient Boosting) to predict optimal CPI.
- **Actionable Recommendations**: Insights to help pricing teams adjust bids for competitiveness and profitability.

---

## 🗂 Folder Structure

```
Final-Version/
├── BidPricingAnalytics/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── insights.py
│   │   ├── overview.py
│   │   ├── prediction.py
│   │   └── analysis/
│   │       ├── __init__.py
│   │       ├── analysis_basic.py
│   │       └── analysis_advanced.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_processor.py
│   │   ├── visualization.py
│   │   ├── visualization_analysis.py
│   │   ├── visualization_basic.py
│   │   └── visualization_prediction.py
│   ├── config.py              # Centralized app configuration
│   └── main.py                # Streamlit app entry point
├── data/
│   ├── Account+List+with+Segment.csv
│   ├── Data Dictionary.xlsx
│   ├── DealItemReportLOST.xlsx
│   └── invoiced_jobs_this_year_*.xlsx
├── docs/
│   ├── installation-guide.txt
│   └── readme-file.txt
├── .gitignore
├── README.md
└── requirements.txt

```

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yatharthk2/PredictiveDashboarding.git
cd PredictiveDashboarding

# (Optional) Create virtual environment
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/fixed-app-updated.py
```

---

## 📁 Data Sources

- `invoiced_jobs_*.xlsx` – Deals that were successfully invoiced
- `DealItemReportLOST.xlsx` – Lost or unconverted bids

---

## 🧠 Models Used

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

All trained on engineered features like `IR/LOI` ratio, `IR/Completes`, and categorical `Type`.

---

## 🙋‍♂️ Author

Developed by **Shriyansh Singh** – [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)

---

## 📃 License

This project is for educational and demonstration purposes only.

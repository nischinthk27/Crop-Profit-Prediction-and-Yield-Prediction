### Crop-Profit-Prediction-and-Yield-Prediction
🌾 ML-powered web app that predicts crop profit using yield &amp; price data to help farmers make smarter decisions.

A web-based agricultural decision support system that predicts crop profitability using historical yield and market price data. Built to help farmers and agricultural planners make data-driven crop selection decisions.

## 📌 Problem Statement
Farmers often lack reliable tools to predict crop profitability before cultivation. Traditional methods rely on guesswork and past experience, ignoring price trends and regional variations — leading to poor crop selection and financial losses.

## 🎯 Objectives

Analyze historical agricultural data (crop yield + price trends)
Preprocess data using encoding and cleaning techniques
Train and compare multiple ML regression models
Deploy a user-friendly web-based prediction system

## 🗂️ Dataset

| Dataset | Description |
|---|---|
| Crop Yield Enriched Dataset | Crop type, district, production, yield stats |
| Crop Price Trends Dataset | Historical market prices (2015–2025) |

- **Total Records:** 19,000+
- **Source:** Karnataka Agriculture (data.gov.in)

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Libraries | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Web Framework | Flask |
| Model Saving | Joblib |
| Frontend | HTML, CSS |

## 🤖 Models Used

Linear Regression
Decision Tree Regressor
Random Forest Regressor ✅
XGBoost Regressor ✅ (best performing)

## 📊 Results

| Metric | Value |
|---|---|
| R² Score | ~0.97 |
| Best Model | XGBoost / Random Forest |
| Mean AP (Yield Classification) | 0.9293 |

## 🔄 Workflow
Data Source → Data Collection → Preprocessing → Feature Engineering
     → Model Training → Model Evaluation → Prediction & Output

## 🖥️ Web Interface
Users can input:

Season, Crop, District
Year, Area (acres)
Previous year production, area, and total cost

The app returns predicted yield, market price, estimated profit, and ROI.

## 🚀 How to Run
# Clone the repository
git clone https://github.com/Nischinth/Crop-Profit-Prediction.git
cd Crop-Profit-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

# 📁 Project Structure
```
Crop-Profit-Prediction/
│
├── app.py                  # Flask application
├── model/                  # Saved ML models (.pkl)
├── templates/              # HTML templates
├── static/                 # CSS files
├── datasets/               # CSV datasets
├── notebooks/              # EDA and model training
└── requirements.txt

## 🔭 Future Scope

Integration with real-time weather APIs and IoT soil sensors
Mobile app version for rural accessibility
Multilingual interface support
Crop recommendation and fertilizer optimization features


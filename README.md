# 📊 Stock Market Analysis & Prediction

## 📌 Project Overview
This project analyzes historical stock price data for **AAPL**, **MSFT**, **NFLX**, and **GOOG** to identify trends, correlations, volatility, and make **next-day closing price predictions** using **Machine Learning**.

**Tools Used:** Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Excel, SQL (optional).

---

## 🎯 Objectives
- Analyze historical stock data and identify **trends & patterns**.
- Calculate **moving averages** and **volatility**.
- Perform **correlation analysis**.
- Build a **Linear Regression** model to predict the next day’s closing price.

---

## 📂 Dataset
- **Rows:** 248
- **Columns:** `Ticker`, `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- No missing values after cleaning.

---

## 🛠 Steps Performed
### 1️⃣ Data Cleaning
- Converted `Date` to datetime format.
- Converted price & volume columns to numeric.
- Removed invalid rows (NaN).
- Sorted by `Ticker` & `Date`.

### 2️⃣ Feature Engineering
- Added **Returns** (`% change` in Close price).
- Added **MA(7)** & **MA(30)** (moving averages).
- Added **Volatility(7)** & **Volatility(30)**.

### 3️⃣ Exploratory Data Analysis (EDA)
- Summary statistics.
- Correlation matrix heatmap.
- Close price trends by ticker.
- Total traded volume by ticker.

### 4️⃣ Machine Learning
- **Model:** Linear Regression
- **Target:** Next day’s `Close` price
- **Features:** Close, MA(7), MA(30), Volatility(7), Volatility(30), Return
- **Metrics:** RMSE, MAE

---

## 📊 Key Insights
- Price columns are **highly correlated** (~1.00).
- Volume has a **negative correlation** (~ -0.54) with prices.
- MSFT had the highest traded volume; NFLX the lowest.
- MSFT & AAPL showed stable movements, while NFLX & GOOG were more volatile.
- Linear Regression provided reasonable prediction accuracy.

---

## 📈 Visualizations
- Correlation Matrix Heatmap
- Close Price Trends Over Time
- Total Volume by Ticker
- Moving Averages (7 & 30 days)
- Predictions vs Actual Close Price

---

## 📦 Files in This Repository
- `stock.py` → Main analysis & prediction script.
- `stocks.csv` → Dataset.
- `stock_market.docx` → Project report with findings and screenshots.


---

## 🚀 Next Steps
- Try alternative models like **ARIMA** or **Facebook Prophet** for time series forecasting.
- Implement **hyperparameter tuning** and **cross-validation**.
- Deploy as a **web app** using Flask or Django.

---

## 📜 License
This project is licensed under the **MIT License** – feel free to use and modify.

---

## 👩‍💻 Author
**Isha Koolwal**  
Data Analyst & Machine Learning Enthusiast

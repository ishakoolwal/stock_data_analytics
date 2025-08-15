# stock.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


file_path = "C:/Users/MSI/Downloads/stocks.csv" 
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

num_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=['Date'] + num_cols, inplace=True)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

print("\nAfter cleaning:")
print(df.info())


print("\nUnique tickers:", df['Ticker'].unique())
print("\nStatistical summary:")
print(df.describe())


corr = df[num_cols].corr()
print("\nCorrelation matrix:")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()
plt.close()


plt.figure(figsize=(12, 6))
for t in df['Ticker'].unique():
    sub = df[df['Ticker'] == t]
    plt.plot(sub['Date'], sub['Close'], label=t)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Close Price Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("close_price_trends.png")
plt.show()
plt.close()


plt.figure(figsize=(6, 4))
total_volume = df.groupby('Ticker')['Volume'].sum()
sns.barplot(x=total_volume.index, y=total_volume.values)
plt.title("Total Volume by Ticker")
plt.ylabel("Total Volume")
plt.tight_layout()
plt.savefig("total_volume.png")
plt.show()
plt.close()


def add_indicators(group):
    group = group.sort_values('Date')
    group['Return'] = group['Close'].pct_change()
    group['MA_7'] = group['Close'].rolling(7).mean()
    group['MA_30'] = group['Close'].rolling(30).mean()
    group['Volatility_7'] = group['Close'].rolling(7).std()
    group['Volatility_30'] = group['Close'].rolling(30).std()
    return group

df = df.groupby('Ticker', group_keys=False)[df.columns].apply(add_indicators).reset_index(drop=True)


for t in df['Ticker'].unique():
    sub = df[df['Ticker'] == t]
    plt.figure(figsize=(10, 5))
    plt.plot(sub['Date'], sub['Close'], label='Close')
    plt.plot(sub['Date'], sub['MA_7'], label='MA 7', linestyle='--')
    plt.plot(sub['Date'], sub['MA_30'], label='MA 30', linestyle='--')
    plt.title(f"{t} Close Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{t.lower()}_moving_averages.png")
    plt.show()
    plt.close()


results = {}
for t in df['Ticker'].unique():
    sub = df[df['Ticker'] == t].sort_values('Date').copy()
    
    # Target = Next day's close
    sub['Target'] = sub['Close'].shift(-1)
    
    # Features for prediction
    features = ['Close', 'MA_7', 'MA_30', 'Volatility_7', 'Volatility_30', 'Return']
    sub = sub.dropna(subset=features + ['Target'])
    
    X = sub[features]
    y = sub['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    
    results[t] = {"RMSE": rmse, "MAE": mae}
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, preds, label="Predicted", linestyle='--')
    plt.title(f"{t} — Next Day Close Price Prediction")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{t.lower()}_predictions.png")
    plt.show()
    plt.close()

metrics_df = pd.DataFrame(results).T
metrics_df.to_csv("model_metrics.csv")
print("\nModel performance metrics:")
print(metrics_df)


df.to_csv("cleaned_stocks.csv", index=False)


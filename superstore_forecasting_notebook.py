# superstore_forecasting_notebook.py
# Requirements: pandas, numpy, matplotlib, seaborn, prophet (or fbprophet), statsmodels, scikit-learn

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import numpy as np

# For Prophet (install as 'prophet' or 'fbprophet' depending on pip)
try:
    from prophet import Prophet
except Exception as e:
    print("Prophet not available. Install with: pip install prophet")
    Prophet = None

import statsmodels.api as sm

# ---------- 1. Load ----------
DATA_PATH = "data:superstore_sales.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['Order Date','Ship Date'], dayfirst=False, encoding='latin1', low_memory=False)
print("Initial shape:", df.shape)
df.head()

# ---------- 2. Clean ----------
# Standardize column names
df.columns = [c.strip() for c in df.columns]
# Drop rows missing order date or sales
df = df.dropna(subset=['Order Date', 'Sales'])
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

# ---------- 3. ETL / KPI ----------
# Create month, quarter
df['order_month'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
daily = df.set_index('Order Date').resample('D').agg({'Sales':'sum','Quantity':'sum'})
monthly = df.set_index('Order Date').resample('MS').agg({'Sales':'sum','Quantity':'sum'})
monthly.index = monthly.index.to_period('M').to_timestamp()

# Top categories
top_categories = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print("Top categories:\n", top_categories)

# ---------- 4. Save aggregated for BI ----------
os.makedirs("outputs", exist_ok=True)
monthly.reset_index().to_csv("outputs/monthly_sales.csv", index=False)
print("Saved outputs/monthly_sales.csv")

# ---------- 5. Forecasting - example on global monthly sales ----------
ts = monthly['Sales'].reset_index().rename(columns={'Order Date':'ds','Sales':'y'})

# Fill missing months by reindexing monthly range
full_idx = pd.date_range(start=ts['ds'].min(), end=ts['ds'].max(), freq='MS')
ts = ts.set_index('ds').reindex(full_idx).fillna(0).rename_axis('ds').reset_index()

# Prophet
if Prophet is not None:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(ts)
    future = m.make_future_dataframe(periods=6, freq='MS')
    fcst = m.predict(future)
    forecast_df = fcst[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds')
    # Evaluate on last N months if you want (train/test split)
    print(forecast_df.tail())
    # Plot
    fig = m.plot(fcst)
    plt.title("Prophet Forecast - Monthly Sales")
    plt.show()

# Statsmodels - simple SARIMAX
y = ts['y']
y.index = ts['ds']
# Use simple seasonal order (p,d,q)x(P,D,Q,s)
sarimax_mod = sm.tsa.statespace.SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
sarimax_res = sarimax_mod.fit(disp=False)
pred = sarimax_res.get_forecast(steps=6)
pred_ci = pred.conf_int()
pred_mean = pred.predicted_mean
# Combine into DataFrame
pred_df = pd.DataFrame({'ds': pd.date_range(start=y.index.max()+pd.offsets.MonthBegin(1), periods=6, freq='MS'),
                        'yhat': pred_mean.values,
                        'yhat_lower': pred_ci.iloc[:,0].values,
                        'yhat_upper': pred_ci.iloc[:,1].values})
print(pred_df)

# ---------- 6. Evaluate (example backtest MAPE) ----------
# Backtest: train on all but last 6 months
train = y.iloc[:-6]
test = y.iloc[-6:]
sarimax_train = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
pred_bt = sarimax_train.get_forecast(steps=6).predicted_mean
mape = mean_absolute_percentage_error(test, pred_bt)
rmse = root_mean_squared_error(test, pred_bt)
print(f"Backtest MAPE: {mape:.4f}, RMSE: {rmse:.2f}")

# Save forecasts
pred_df.to_csv("outputs/monthly_sales_forecast_sarimax.csv", index=False)
if Prophet is not None:
    forecast_df.reset_index().to_csv("outputs/monthly_sales_forecast_prophet.csv")
print("Saved forecast outputs to outputs/")

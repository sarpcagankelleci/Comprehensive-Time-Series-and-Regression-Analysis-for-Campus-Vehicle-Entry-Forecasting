#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm # for residual analysis
import scipy.stats as stats 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import bbplot
from bbplot import bijan
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[4]:


df = pd.read_csv('renault.csv', skiprows=0, header=None)
df = df.iloc[:, :2]
df.columns = ['Date', 'Sales']
df['Date'] = pd.to_datetime(df['Date'])
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
Date = df['Date']
Sales = df['Sales']


# # Q1A
# 

# In[5]:


seasonal_sales = [0] * len(df)

for i in range(12, len(df)):
    seasonal_sales[i] = df['Sales'].iloc[i] - df['Sales'].iloc[i - 12]
seasonal_sales = np.diff(seasonal_sales)


plt.figure(figsize=(12, 6))
plot_acf(seasonal_sales, lags=20, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# Plot PACF
plt.figure(figsize=(12, 6))
plot_pacf(seasonal_sales, lags=20, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()


# # Q1B

# In[6]:


training_data = df['Sales'][:96]
testing_data =df['Sales'][96:]
print(training_data)
print(testing_data)


# # Q1C

# In[7]:



Arima_model_train = ARIMA(training_data, order=(0, 0, 1), seasonal_order=(0, 0, 0, 12))
Arima_model_fit = Arima_model_train.fit()

# Print model summary
print(Arima_model_fit.summary())

# Get statistical significance of coefficients
coef_pvalues = Arima_model_fit.pvalues
print("\nStatistical significance of coefficients:")
print(coef_pvalues)

# Report AIC
AIC = Arima_model_fit.aic
print("\nAIC:", AIC)


# # Q1D

# In[10]:



start_index = len(training_data)
end_index = start_index + len(testing_data) - 1

predict_test = Arima_model_fit.predict(start=start_index, end=end_index, typ='levels')


Arima_model_test= ARIMA(testing_data, order=(0, 0, 1), seasonal_order=(0, 0, 0, 12))
Arima_model_test_fit = Arima_model_test.filter(Arima_model_fit.params)


predict_test = Arima_model_fit.get_forecast(steps=len(testing_data))
predict_test_mean = predict_test.predicted_mean

# Calculate errors using the predicted mean and the actual test data
mse_test = mean_squared_error(testing_data, predict_test_mean)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(testing_data, predict_test_mean)
mape_test = np.mean(np.abs(predict_test_mean - testing_data) / testing_data)

print('\nBenchmark Model Evaluation on Test Data')
print('MSE for testing data =', mse_test)
print('RMSE for testing data =', rmse_test)
print('MAE for testing data =', mae_test)
print('MAPE for testing data =', mape_test)


# # Q1E

# In[11]:


Arima_model_train_new = ARIMA(training_data, order=(0, 0, 1), seasonal_order=(0, 1, 1, 12))
Arima_new = Arima_model_train_new.fit()

# Print model summary
print(Arima_new.summary())

# Get statistical significance of coefficients
coef_pvalues = Arima_new.pvalues
print("\nStatistical significance of coefficients:")
print(coef_pvalues)

# Report AIC
AIC = Arima_new.aic
print("\nAIC:", AIC)


# # Q1F

# In[12]:


Arima_model_test_new= ARIMA(testing_data, order=(0, 0, 1), seasonal_order=(0, 1, 1, 12))
Arima_model_test_new = Arima_model_test_new.filter(Arima_new.params)

# Q1F - Evaluate Alternative ARIMA Model on Test Data
# Correct way to make out-of-sample predictions
predict_test_new = Arima_new.get_forecast(steps=len(testing_data))
predict_test_new_mean = predict_test_new.predicted_mean

# Calculate errors using the predicted mean and the actual test data
mse_test_new = mean_squared_error(testing_data, predict_test_new_mean)
rmse_test_new = np.sqrt(mse_test_new)
mae_test_new = mean_absolute_error(testing_data, predict_test_new_mean)
mape_test_new = np.mean(np.abs(predict_test_new_mean - testing_data) / testing_data)

print('\nAlternative Model Evaluation on Test Data')
print('MSE for testing data =', mse_test_new)
print('RMSE for testing data =', rmse_test_new)
print('MAE for testing data =', mae_test_new)
print('MAPE for testing data =', mape_test_new)


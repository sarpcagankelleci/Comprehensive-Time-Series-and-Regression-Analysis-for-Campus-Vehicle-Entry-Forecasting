#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm 
import scipy.stats as stats 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import bbplot
from bbplot import bijan
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[50]:


df = pd.read_csv('vehiclecount.csv', skiprows=0, header=None)
df = df.iloc[:, :2] 
df.columns = ['Date', 'Cars']
df['Date'] = pd.to_datetime(df['Date'])
df['Cars'] = pd.to_numeric(df['Cars'], errors='coerce')
Date = df['Date']
Cars = df['Cars']


# # Q2A

# In[51]:



average_cars = df['Cars'].mean()

plt.figure(figsize=(10, 6))  
plt.plot(df['Date'], df['Cars'], linestyle='-')  

plt.axhline(y=average_cars, color='r', linestyle='--', label='Average')

plt.title('Monthly Sales')  
plt.xlabel('Date')  
plt.ylabel('Cars') 
plt.xticks(rotation=45)  
plt.tight_layout() 

plt.legend()

plt.show()


# # Q2B

# In[52]:


Q1 = df['Cars'].quantile(0.25)
Q3 = df['Cars'].quantile(0.75)
IQR = Q3 - Q1
print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Outlier'] = (df['Cars'] < lower_bound) | (df['Cars'] > upper_bound)

df_no_outliers = df[~df['Outlier']]

print("Original data with outliers designated:")
print(df)

print("\nData with outliers removed:")
print(df_no_outliers)


# THERE APPEARS TO BE NO EXTREME OUTLIERS.

# # Q2C

# In[53]:


training_data = df['Cars'][:122]
testing_data =df['Cars'][122:]
print("Training data:")
print(training_data, "\n")
print("Testing data:")
print(testing_data)


# # Q2D

# In[54]:



df = pd.read_csv('vehiclecount.csv', skiprows=0, header=None)
df = df.iloc[:, :2] 
df.columns = ['Date', 'Cars']
df['Date'] = pd.to_datetime(df['Date'])
df['Cars'] = pd.to_numeric(df['Cars'], errors='coerce')

df['Forecast'] = df['Cars'].shift(7)
print(df)

plt.figure(figsize=(10, 6))

plt.plot(df.index, df['Cars'], label='Actual Cars', linestyle='-')

plt.plot(df.index[7:], df['Forecast'][7:], label='Forecast', linestyle='--')

plt.title('Daily Cars Forecast')  
plt.xlabel('Date')  
plt.ylabel('Cars') 
plt.xticks(rotation=45)  
plt.legend()
plt.tight_layout() 

plt.show()


# In[55]:


train_actual = df['Cars'][:130]
train_forecast = df['Forecast'][:130].dropna()
train_rmse = np.sqrt(mean_squared_error(train_actual[7:], train_forecast))
train_mape = np.mean(np.abs((train_actual[7:] - train_forecast) / train_actual[7:])) * 100

test_actual = df['Cars'][130:]
test_forecast = df['Forecast'][130:].dropna()
test_rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
test_mape = np.mean(np.abs((test_actual - test_forecast) / test_actual)) * 100

print("Training RMSE:", train_rmse)
print("Training MAPE:", train_mape)
print("Testing RMSE:", test_rmse)
print("Testing MAPE:", test_mape)


# # Q5E

# In[56]:


differenced_data=df["Cars"].values[7:] - df["Forecast"].values[7:]
plt.plot(differenced_data)


# In[57]:


fig, axes = plt.subplots(1, 2, figsize=(20,4))
fig1 = sm.graphics.tsa.plot_acf(differenced_data, lags=30, ax=axes[0])
fig2 = sm.graphics.tsa.plot_pacf(differenced_data, lags=30, ax=axes[1])


# In[58]:


sarima_training = sm.tsa.statespace.SARIMAX(training_data, trend ='c',  order=(1,0,0) , seasonal_order=(0,1,1,7))
rb_sarima_training = sarima_training.fit(disp=False)
rb_sarima_training.summary()


# # Q2F

# In[59]:


predicted_training = rb_sarima_training.predict()

mae_training = mean_absolute_error(training_data, predicted_training)
mape_training = np.mean(np.abs((training_data - predicted_training) / training_data)) * 100
rmse_training = np.sqrt(mean_squared_error(training_data, predicted_training))

print("MAE on training data:", mae_training)
print("MAPE on training data:", mape_training)
print("RMSE on training data:", rmse_training)


# # Q2G

# In[60]:


sarima_testing = sm.tsa.statespace.SARIMAX(testing_data, trend ='c',  order=(1,0,0) , seasonal_order=(0,1,1,7))
rb_sarima_testing = sarima_testing.fit(disp=False)
rb_sarima_testing.summary()


# In[61]:


predicted_testing = rb_sarima_testing.predict()

mae_testing= mean_absolute_error(testing_data, predicted_testing)
mape_testing = np.mean(np.abs((testing_data - predicted_testing) / testing_data)) * 100
rmse_testing = np.sqrt(mean_squared_error(testing_data, predicted_testing))

print("MAE on training data:", mae_testing)
print("MAPE on training data:", mape_testing)
print("RMSE on training data:", rmse_testing)


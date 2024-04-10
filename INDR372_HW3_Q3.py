#!/usr/bin/env python
# coding: utf-8

# In[184]:


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from statsmodels.api import OLS, add_constant


# #PART A & B

# In[185]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
vehicle_count_days_df = pd.read_csv('vehiclecountdays.csv')

# Convert 'DATE' to datetime and sort the dataframe by date
vehicle_count_days_df['DATE'] = pd.to_datetime(vehicle_count_days_df['DATE'])
vehicle_count_days_df.sort_values(by='DATE', inplace=True)

# Debugging print statement
print(vehicle_count_days_df.head())

# Prepare predictors
vehicle_count_days_df['Trend'] = np.arange(len(vehicle_count_days_df)) + 1
vehicle_count_days_df['Trend_squared'] = vehicle_count_days_df['Trend'] ** 2

# Add week of the month
vehicle_count_days_df['Week_of_month'] = vehicle_count_days_df['DATE'].apply(lambda x: (x.day-1) // 7 + 1)
weeks_of_month_dummies = pd.get_dummies(vehicle_count_days_df['Week_of_month'], prefix='Week')
vehicle_count_days_df = pd.concat([vehicle_count_days_df, weeks_of_month_dummies], axis=1)

# Add lagged variables
vehicle_count_days_df['Lag_7'] = vehicle_count_days_df['Number of Vehicles'].shift(7)
vehicle_count_days_df['Lag_14'] = vehicle_count_days_df['Number of Vehicles'].shift(14)

# Drop rows with NaN values (due to lagged variables)
vehicle_count_days_df.dropna(inplace=True)

# Define predictors and target variable
X = vehicle_count_days_df.drop(['Number of Vehicles', 'DATE', 'Week_of_month'], axis=1)
y = vehicle_count_days_df['Number of Vehicles']

# Split data into training and testing sets
split_index = int(len(X) * 0.75)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


# Debugging print statements
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

print("\n", X_train)
print("\n", y_train)


# #PART C

# In[186]:


#The predictors we're using are:

#Trend: Linear time trend.
#Trend_squared: Quadratic time trend.
#Day of the Week Dummies: Binary variables for each day of the week (Sunday through Friday; Saturday can be inferred).
#Weeks of the Month Dummies ('Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5'): Binary variables for the first four weeks of the month (Week 5 can be inferred).
#Lag_7: Vehicle count lagged by 7 days.
#Lag_14: Vehicle count lagged by 14 days.


# #PART D

# In[187]:


# Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on training set
y_pred_train = model.predict(X_train)

# Calculate R^2 and Adjusted R^2
r2 = r2_score(y_train, y_pred_train)
adj_r2 = 1 - (1-r2)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)

# Using OLS for AIC
X_train_const = add_constant(X_train)
ols_model = OLS(y_train, X_train_const).fit()

print(f"R^2: {r2}, Adjusted R^2: {adj_r2}, AIC: {ols_model.aic}")


# #PART E

# In[188]:


# Calculate RMSE and MAPE for the training data
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mape_train = mean_absolute_percentage_error(y_train, y_pred_train)

print(f"RMSE (Training): {rmse_train}, MAPE (Training): {mape_train}%")


# #PART F

# In[189]:


print(ols_model.summary())


# #PART G

# In[190]:


# Predictions on the test set
y_pred_test = model.predict(X_test)

# Calculate RMSE and MAPE for the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"RMSE (Test): {rmse_test}, MAPE (Test): {mape_test}%")


# #PART H

# In[191]:


# Redefine predictors for the reduced model
X_reduced = X_train.drop(['Trend_squared', 'Week_4'], axis=1)
X_test_reduced = X_test.drop(['Trend_squared', 'Week_4'], axis=1)

# Refit model with reduced predictors
model_reduced = LinearRegression()
model_reduced.fit(X_reduced, y_train)

# New OLS model for reduced predictors
X_reduced_const = add_constant(X_reduced)
ols_model_reduced = OLS(y_train, X_reduced_const).fit()

print(ols_model_reduced.summary())


# #PART I

# In[192]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the data
vehicle_count_days_df = pd.read_csv('vehiclecountdays.csv')

# Convert 'DATE' to datetime and sort the dataframe by date
vehicle_count_days_df['DATE'] = pd.to_datetime(vehicle_count_days_df['DATE'])
vehicle_count_days_df.sort_values(by='DATE', inplace=True)

# Prepare predictors
vehicle_count_days_df['Trend'] = np.arange(len(vehicle_count_days_df)) + 1
vehicle_count_days_df['Trend_squared'] = vehicle_count_days_df['Trend'] ** 2
vehicle_count_days_df['Week_of_month'] = vehicle_count_days_df['DATE'].apply(lambda x: (x.day-1) // 7 + 1)
weeks_of_month_dummies = pd.get_dummies(vehicle_count_days_df['Week_of_month'], prefix='Week')
vehicle_count_days_df = pd.concat([vehicle_count_days_df, weeks_of_month_dummies], axis=1)
vehicle_count_days_df['Lag_7'] = vehicle_count_days_df['Number of Vehicles'].shift(7)
vehicle_count_days_df['Lag_14'] = vehicle_count_days_df['Number of Vehicles'].shift(14)
vehicle_count_days_df.dropna(inplace=True)  # Drop rows with NaN values created by lagged variables

# Corrected Full Model Predictors
full_predictors = ['Trend', 'Trend_squared', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 
                   'Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Lag_7', 'Lag_14']

# First Reduced Model Predictors
reduced_predictors_1 = ['Trend', 'Lag_7', 'Lag_14', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Second Reduced Model Predictors
reduced_predictors_2 = ['Trend', 'Lag_7', 'Sunday', 'Monday']

# Split the dataset into training and test sets
X = vehicle_count_days_df[full_predictors]
y = vehicle_count_days_df['Number of Vehicles']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Fit the full model
full_model = LinearRegression().fit(X_train, y_train)

# Fit the first reduced model
reduced_model_1 = LinearRegression().fit(X_train[reduced_predictors_1], y_train)

# Fit the second reduced model
reduced_model_2 = LinearRegression().fit(X_train[reduced_predictors_2], y_train)

# Evaluate the full model
full_pred = full_model.predict(X_test)
full_rmse = np.sqrt(mean_squared_error(y_test, full_pred))
full_mape = mean_absolute_percentage_error(y_test, full_pred) * 100

# Evaluate the first reduced model
reduced_pred_1 = reduced_model_1.predict(X_test[reduced_predictors_1])
reduced_rmse_1 = np.sqrt(mean_squared_error(y_test, reduced_pred_1))
reduced_mape_1 = mean_absolute_percentage_error(y_test, reduced_pred_1) * 100

# Evaluate the second reduced model
reduced_pred_2 = reduced_model_2.predict(X_test[reduced_predictors_2])
reduced_rmse_2 = np.sqrt(mean_squared_error(y_test, reduced_pred_2))
reduced_mape_2 = mean_absolute_percentage_error(y_test, reduced_pred_2) * 100

# Output the results
print("Full Model RMSE:", full_rmse, "MAPE:", full_mape)
print("Reduced Model 1 RMSE:", reduced_rmse_1, "MAPE:", reduced_mape_1)
print("Reduced Model 2 RMSE:", reduced_rmse_2, "MAPE:", reduced_mape_2)


# #PART J

# In[176]:


# This step involves summarizing the analysis, including the rationale for choosing predictors, the significance of each predictor, the performance of the initial and reduced models, and recommendations for forecasting the number of vehicles.

#While the report's specifics would depend on the outcomes of the above steps, the structure might include:

#Introduction: Briefly describe the objective and approach.
#Model Development: Explain the initial model, including predictor selection and performance metrics.
#Model Refinement: Discuss the rationale for reducing the model, the process, and the outcomes.
#Conclusion and Recommendations: Summarize the findings, state the preferred model, and provide forecasting recommendations based on the analysis.
#This comprehensive approach, from data preparation through model refinement and reporting, encompasses a thorough analysis process for predicting the number of vehicles entering a campus, adhering to the steps outlined in parts (a) through (j).


# @Credit - Sarp Çağan Kelleci & Tan Karahasanoğlu

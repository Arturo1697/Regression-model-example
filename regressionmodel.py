#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt	
import pandas as pd
import seaborn as sns
import time

start_time = time.time()
#upload dataset
dataset = pd.read_csv('/home/arturo/Downloads/1000_Companies.csv')
#split dataset
#: take every row except last column
x = dataset.iloc[:, :-1].values 
# set equal to the last row
y = dataset.iloc[:, 4].values
# .head() first 5 rows of data
#print(dataset.head())

#corr for coordinates
#sns.heatmap(dataset.corr())

#preparation
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#Label encoder switches labels
labelencoder = LabelEncoder()

#editing only the thrid column into numbers
x[:, 3] = labelencoder.fit_transform(x[:, 3])

#ct = ColumnTransformer([("State",OneHotEncoder(),[3])],remainder = 'passthrough')
#x=ct.fit_transform(x)
#column converts the states number 2,1,0 into 0.0,0.0,0.0 format due to the chance of using the greater number when applying through regression model

# removes that extra column/ Avoiding dummy variable trap by removing the first column of 0.0 it can still detect that 1st class by viewing 0.0 0.0 meaning the the first column was a 1.0
#x = x[:, 1:]



#Linear regression
# Splitting the dataset into the Training set and Test set 0.2=20% 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

#print(regressor.coef_)
#print(regressor.intercept_)

# Calculating the R squared value
from sklearn.metrics import r2_score
Z=r2_score(y_test, y_pred)


plt.scatter(x_test[:,0], y_test, color="black")
plt.plot(x_test[:,0],y_pred, color = "red",linewidth=1,label= "prediction line")
plt.xlabel('R&D spend')
plt.ylabel('Profit')
plt.legend()
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))




























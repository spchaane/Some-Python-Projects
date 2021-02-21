#!/usr/bin/env python
# coding: utf-8

# **Explanatory Linear Regression Function**
# 
# Explonatory Data Analysis to check how the Open, High, Low and Volume affects the closing price fot the GOOGL stock market

# In[56]:


#import necessary modules
import pandas as pd
import numpy as np


# In[57]:


#read the usdcad.csv file and convert it into a dataframe
df = pd.read_csv("GOOGL.csv", header = 0, sep = ",")

#first 5 entries of the df
df.head()


# Check the descriptive stats and info on the data

# In[58]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print("Descriptive Statistics\n", df.describe()) #descriptive stat on data


# In[59]:


print("\nInformation on Data\n", df.info()) #prints the information on data


# **Data Cleaning**

# In[60]:


#check columns with missing values
columns_missing_values = df.isna().sum() #sums columns with missing values
#filter columns with at least one missing value
columns_missing_values = columns_missing_values[columns_missing_values != 0]
#print number of columns with missing values
print("Total Columns Missing Values: ", len(columns_missing_values))


# There are no columns that has any missing values. Therefore we continue with the analysis

# **Correlation Matrix**
# 
# Check Correlation within the variables to see which one has a positive impact on the closing price

# In[61]:


#creates a correlation matrix
print("Correlation Matrix")
round(df.corr(), 2)


# The correlation is high between closing, high, low, open and adj closing price. Meaning the high, low and opening price have
# an impact on the closing price. Therefore we can use any of the variables as explanatory variables to predict the closing price or all of them.

# **Visualizing the Correlation**
# 
# Heatmap

# In[62]:


#import seaborn and matplotlib.pyplot to be able to plot the heatmap
import seaborn as sns
import matplotlib.pyplot as plt


# In[63]:


correlation_df = df.corr() #assign a correlation_df to correlation data

axis_corr = sns.heatmap(correlation_df, vmin = -1, vmax = 1, center = 0, cmap = sns.diverging_palette(50, 500, n = 500), square = True)
plt.show()


# Scatter Graph

# In[64]:


#plots scatter graph between opening and closing prices
df.plot(x = "Open", y = "Close", kind = "scatter"),
plt.ylabel("Closing Price")
plt.xlabel("Opening Price")
plt.show()


# In[66]:


#Since there is no relationship between the Volumes and the Closing price (indicated by a negative correlation), we are removing the Date Column
df.drop(columns = ["Volume"], inplace = True)

print(df.head(3))


# **Linear Regression Table**
# 
# Build a Linear Regression table using the High, Low, open and close

# In[67]:


#import the necessary modules
import statsmodels.formula.api as smf


model = smf.ols("Close ~ Open + High + Low", data = df)
results = model.fit() #fit the data properly
print(results.summary()) #prints the summary of results


# The R-Squared of the explanatory variables and the variable we want to predict is high at 0.997, meaning most of the data points are close to the linear regression function line, therefore the model fts the data very well.
# 
# The P-Value of the Intercept, Open, High and Low is below 0.050, which means that the three vaiables have a significant impact on the closing price of the GOOGL. Therefore we can reject the null hypothesis
# 
# The coefficient indicates that the closing price will increas by -0.4936, 0.7042, 0.7833 when the Open, High and Low increases by 1 respectively. The intercept of the Closing price is 11.3424 when the Open, High and Low are 0.

# **Creating the Function**
# 
# Creating the function with the data from the OLS Regression Table

# In[70]:


df.tail(2)


# In[72]:


#creating the prediction function using the open, high and low prices
def predict_closing_price(Open, High, Low):
    return Open * -0.4936 + High * 0.7042 + Low * 0.7833 + 11.3424

#use the last open, high and low to predict the closing price
print("Close: ",predict_closing_price(2107.790039, 2118.580078, 2096.850098))


# In[73]:


#Use Open price to predict the Close price
def predict_close(Open):
    return Open * -0.4936 + 11.3424

#use the Open price to predict the closing price
print("Close: ", predict_close(2107.790039))


# In[78]:


#Use High price to predict the Close price
def predict_close(High):
    return High * 0.7042 + 11.3424

#use the High price to predict the closing price
print("Close: ", predict_close(2118.580078))


# **Creating the Linear Regression Model**

# In[79]:


#import the necessary library
from scipy import stats

#Create the model
x = df["Open"]
y = df["Close"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myFunc(x):
    return slope * x + intercept

mymodel = list(map(myFunc, x))

plt.scatter(x, y) #plots an ordinary scatter graph
plt.plot(x, mymodel) #plots a linear regression model
plt.ylim(1000, 2200)
plt.xlim(1000, 2200)
plt.show()


# In[ ]:





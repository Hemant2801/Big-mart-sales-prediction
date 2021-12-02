#!/usr/bin/env python
# coding: utf-8

# # Importing all the dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


# # Data collection and analysis

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Big mart sales prediction/train.csv')


# In[3]:


#print the fisrt 5 rows of the dataset
'''
FD = food 
DR = drink
NC = non consumable

'''
df.head()


# In[4]:


# print the last 5 rows of the dataset
df.tail()


# In[5]:


# shape of the dataset
df.shape


# In[6]:


# getting some info about the dataset
df.info()


# In[7]:


#checking for any missing values
df.isnull().sum()


# In[8]:


# stastical measure of the dataset
df.describe()


# In[9]:


#checking for categorical data in diff object type columns
objlist = df.select_dtypes('object').columns
for i in objlist:
    print(f'\n{i}')
    print(df[i].value_counts(), end = '\n') 


# Handling the missing values
# 
# Mean ---> Average value
# Mode ---> Most repeated value

# In[10]:


# mean value of 'Item weight' collumn
mean_value = df['Item_Weight'].mean()


# In[11]:


# filling the missing value with mean in 'item weight' column
df['Item_Weight'].fillna(mean_value, inplace = True)


# In[12]:


#checking for missing values
df.isnull().sum()


# In[13]:


# replacing the missing value with mode in 'Outlet Size' column
mode_value = df.pivot_table(values = 'Outlet_Size', columns = 'Outlet_Type', aggfunc = (lambda x : x.mode()[0]))


# In[14]:


print(mode_value)


# In[15]:


missing_values = df['Outlet_Size'].isnull()


# In[16]:


df.loc[missing_values, 'Outlet_Size'] = df.loc[missing_values, 'Outlet_Type'].apply(lambda x : mode_value[x])


# In[17]:


#checking for missing values
df.isnull().sum()


# Data analysis

# In[18]:


# stastical measure of the data
df.describe()


# Numerical features

# In[19]:


sns.set_style(style = 'darkgrid')


# In[20]:


#item weight distribution
plt.figure(figsize = (6,6))
sns.displot(df['Item_Weight'], kde= True)
plt.show()


# In[21]:


#item visibility distribution
plt.figure(figsize = (6,6))
sns.displot(df['Item_Visibility'], kde= True)
plt.show()


# In[22]:


#item MRP distribution
plt.figure(figsize = (6,6))
sns.displot(df['Item_MRP'], kde= True)
plt.show()


# In[23]:


#Item_Outlet_Sales distribution
plt.figure(figsize = (6,6))
sns.displot(df['Item_Outlet_Sales'], kde= True)
plt.show()


# In[24]:


#Outlet_Establishment_Year distribution
plt.figure(figsize = (6,6))
sns.countplot(x= 'Outlet_Establishment_Year', data = df)
plt.show()


# Categoruical features

# In[25]:


#Item_Fat_Content distribution
plt.figure(figsize = (6,6))
sns.countplot(x= 'Item_Fat_Content', data = df)
plt.show()


# In[26]:


# Item_Type	 distribution
plt.figure(figsize = (30,6))
sns.countplot(x= 'Item_Type', data = df)
plt.show()


# In[27]:


# Outlet location type distribution
plt.figure(figsize = (6,6))
sns.countplot(x = 'Outlet_Location_Type', data = df)
plt.show()


# # Data preprocessing

# In[28]:


df.head()


# In[29]:


df['Item_Fat_Content'].value_counts()


# In[30]:


df.replace({'Item_Fat_Content' : {'low fat' : 'Low Fat', 'LF' : 'Low Fat', 'reg' : 'Regular'}}, inplace = True)


# In[31]:


df['Item_Fat_Content'].value_counts()


# Label Encoding

# In[32]:


encoder = LabelEncoder()

objlist = df.select_dtypes('object').columns
for i in objlist:
    df[i] = encoder.fit_transform(df[i])


# In[33]:


df.head()


# In[34]:


correlation = df.corr()


# In[43]:


plt.figure(figsize = (20,20))
sns.heatmap(correlation , cbar = True, cmap = 'Blues',square = True, annot = True, fmt = '.1f', annot_kws = {'size' : 8})


# Splitting features and targets

# In[36]:


X = df.drop(columns = 'Item_Outlet_Sales' ,axis = 1)
Y = df['Item_Outlet_Sales']


# # Splitting the data into training and testing data

# In[37]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state = 6)


# In[38]:


print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


# # Machine learning model

# In[39]:


model = XGBRegressor()


# In[40]:


model.fit(x_train, y_train)


# Model evaluatuion on training data

# In[41]:


train_prediction = model.predict(x_train)

accuracy_training = metrics.r2_score(y_train, train_prediction)
print('R SQUARED ERROR OF TRAINING DATA :', accuracy_training)


# Model evaluatuion on testing data

# In[42]:


test_prediction = model.predict(x_test)

accuracy_testing = metrics.r2_score(y_test, test_prediction)
print('R SQUARED ERROR OF TESTING DATA :', accuracy_testing)


# In[ ]:





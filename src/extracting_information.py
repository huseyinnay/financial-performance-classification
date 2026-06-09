#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


financial_data = pd.read_csv("../data/raw/financial_data.csv")


# In[3]:


financial_data.head()


# In[4]:


financial_data.info()


# In[5]:


financial_data.describe()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


plt.figure(figsize=(12, 8))
sns.heatmap(financial_data.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Financial Indicators")
plt.show()


# In[15]:


key_metrics = ['Revenue', 'Net Income', 'Equity', 'Total Debt', 'Debt to Equity Ratio', 'Return on Assets', 'Return on Equity']
for metric in key_metrics:
    plt.figure(figsize=(8, 5))
    sns.histplot(financial_data[metric], bins=50, kde=True)
    plt.title(f"Distribution of {metric}")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.show()


# In[14]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=financial_data[['Debt to Equity Ratio', 'Return on Assets', 'Return on Equity', 'Net Profit Margin', 'EBITDA Margin']])
plt.title("Boxplot of Financial Ratios")
plt.xticks(rotation=45)
plt.show()


# In[13]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=financial_data['Revenue'], y=financial_data['Net Income'])
plt.title("Revenue vs. Net Income")
plt.xlabel("Revenue")
plt.ylabel("Net Income")
plt.show()


# In[17]:


subset = financial_data[['Revenue', 'Net Income', 'EBITDA', 'Total Assets', 'Total Liabilities']]
sns.pairplot(subset)
plt.show()


# In[18]:


selected_features = ['Net Income', 'EBITDA Margin', 'Return on Assets', 'Return on Equity', 'Debt to Equity Ratio', 
                     'Interest Coverage Ratio', 'Operating Cash Flow', 'Free Cash Flow', 'Revenue Growth', 'Current Ratio']

plt.figure(figsize=(10, 6))
sns.heatmap(financial_data[selected_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Selected Financial Features")
plt.show()


# In[19]:


sns.pairplot(financial_data[selected_features])
plt.show()


# In[ ]:





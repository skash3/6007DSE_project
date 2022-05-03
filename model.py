#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as pl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('insurance.csv')


# In[3]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import preprocessing


# In[4]:


le = preprocessing.LabelEncoder()


# In[5]:


le.fit(data["sex"])


# In[6]:


data["Sex"] = le.transform(data["sex"])


# In[7]:


le.fit(data["smoker"])


# In[8]:


data["Smoker"] = le.transform(data["smoker"])


# In[9]:


le.fit(data["region"])


# In[10]:


data["Region"] = le.transform(data["region"])


# In[11]:


import statsmodels.api as sm


# In[13]:


y = data["charges"]
x = data[["age", "bmi", "children", "Sex", "Smoker", "Region"]]


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[15]:


from sklearn import linear_model
linreg = linear_model.LinearRegression()
linreg.fit(X_train, y_train)


# In[16]:


predictions = linreg.predict(X_test)


# In[17]:


linreg.score(X_test,y_test)


# In[18]:


import pickle


# In[19]:


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(linreg, f)


# In[20]:


with open('trained_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)


# In[21]:


model = pickle.load(open('trained_model.pkl','rb'))
print(model.predict([[55, 18, 0, 1, 1, 1]]))


# In[ ]:





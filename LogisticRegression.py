
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

ad_data = pd.read_csv('C:/Users/Amogh/Desktop/advertising.csv')


# In[3]:

ad_data.describe()


# In[4]:

ad_data.head()


# In[5]:

ad_data.columns


# In[6]:

from sklearn.model_selection import train_test_split


# In[7]:

X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
Y = ad_data['Clicked on Ad']


# In[8]:

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 42)


# In[9]:

from sklearn.linear_model import LogisticRegression


# In[10]:

logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[25]:

predictions = logmodel.predict(X_test)


# In[26]:

from sklearn.metrics import classification_report


# In[27]:

print(classification_report(Y_test,predictions))


# In[ ]:




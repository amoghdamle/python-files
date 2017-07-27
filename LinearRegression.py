
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[3]:

USAhousing = pd.read_csv('C:/Users/Amogh/Desktop/USA_Housing.csv')


# In[4]:

USAhousing


# In[5]:

USAhousing.describe()


# In[6]:

USAhousing.columns


# In[7]:

sns.heatmap(USAhousing.corr())


# In[9]:

X =USAhousing[['Avg. Area Income', 'Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
Y = USAhousing['Price']


# In[10]:

from sklearn.model_selection import train_test_split


# In[12]:

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.4,random_state = 101)


# In[13]:

Y_test


# In[14]:

from sklearn.linear_model import LinearRegression


# In[16]:

lm=LinearRegression()


# In[17]:

lm.fit(X_train,Y_train)
print(lm.intercept_)


# In[18]:

lm.coef_


# In[19]:

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns = ['Coefficient'])
coeff_df


# In[20]:

predictions = lm.predict(X_test)


# In[21]:

predictions


# In[22]:

from sklearn import metrics


# In[24]:

print('MSE:', metrics.mean_squared_error(Y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,predictions)))


# In[ ]:




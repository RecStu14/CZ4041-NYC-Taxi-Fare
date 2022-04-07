#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression Model

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


checkpoint5 = pd.read_csv('F:\ML Files\checkpoint-5.csv')


# In[3]:


checkpoint5.head(20)


# In[4]:


cols = 'fare_amount'
label = checkpoint5[[cols]]
label


# In[5]:


# checkpoint5 = updated.drop(columns = ['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'dist', 'time'  ])
del checkpoint5['fare_amount']
del checkpoint5['time']


# In[6]:


checkpoint5.head(20)


# In[7]:


# Splitting Dataset into train and test (20% TEST, 80% TRAIN) 
X_train, X_test, y_train, y_test = train_test_split(checkpoint5, label, test_size = 0.2)


# ### Normalising the values

# In[ ]:


X_sc = StandardScaler()
y_sc = StandardScaler()

X_train = X_sc.fit_transform(X_train)
y_train = y_sc.fit_transform(y_train)


# In[ ]:


regressor = LinearRegression()  
t0 = time.time()
regressor.fit(X_train, y_train)
t1 = time.time()


# In[ ]:


with open('linear_Experiment1.pkl', 'wb') as f:

   pickle.dump(regressor, f)


# In[ ]:


y_pred = regressor.predict(X_sc.transform(X_test))
t2 = time.time()
y_pred = y_sc.inverse_transform(y_pred)


# In[ ]:


y_test = y_test.flatten()
df = pd.DataFrame({'Predicted value': y_pred, 'Real Value': y_test})
df


# In[ ]:


time_train = t1-t0
time_predict = t2-t1

print("Linear Regression Accuracy Score -> ",accuracy_score(y_pred,y_test)*100)
# print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))
report = classification_report(y_test, y_pred, output_dict=True)

saved_report = pd.DataFrame(report).transpose()

saved_report['Mean squared error'] = mean_squared_error(y_test, y_pred)
saved_report['Training Time'] = time_train

# Save Report
print("Saving report...")
saved_report.to_csv("linear_report.csv", index=False)
print("Saved report! ")


# In[ ]:


# with open('Linear_Experiment1.pkl', 'rb') as f:

#    regressor = pickle.load(f)


#!/usr/bin/env python
# coding: utf-8

# ### Test accuracy

# In[ ]:


print("Importing libraries...")
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
print("Imported libraries!")


# In[ ]:


checkpoint5 = pd.read_csv('..\checkpoint-5.csv')


# In[ ]:


print("Load Model...")
with open('Linear_Experiment1.pkl', 'rb') as f:

   regressor = pickle.load(f)
print("Model loaded!")


# In[ ]:


print("Preparing data...")
cols = 'fare_amount'
label = checkpoint5[[cols]]
label


# In[ ]:


del checkpoint5['fare_amount']
del checkpoint5['time']
# Splitting Dataset into train and test (20% TEST, 80% TRAIN) 
X_train, X_test, y_train, y_test = train_test_split(checkpoint5, label, test_size = 0.2)


# In[ ]:


print("Doing prediction...")
t1 = time.time()
y_pred = regressor.predict(X_sc.transform(X_test))
t2 = time.time()
y_pred = y_sc.inverse_transform(y_pred)
print("Finished prediction!")


# In[ ]:


print("Saving Report details...")
df = pd.DataFrame({'Predicted value': y_pred, 'Real Value': y_test})
df.to_csv("predicts.csv", index=False)
# time_train = t1-t0
time_predict = t2-t1

print("Linear Regression Accuracy Score -> ",accuracy_score(y_pred,y_test)*100)
# print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))
report = classification_report(y_test, y_pred, output_dict=True)

saved_report = pd.DataFrame(report).transpose()

saved_report['Mean squared error'] = mean_squared_error(y_test, y_pred)
# saved_report['Training Time'] = time_train

# Save Report
print("Saving report...")
saved_report.to_csv("Linear_report.csv", index=False)
print("Saved report! ")


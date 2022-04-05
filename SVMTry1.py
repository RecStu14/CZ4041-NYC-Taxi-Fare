#!/usr/bin/env python
# coding: utf-8

# ## SVM Model

# In[2]:


print("Importing Libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import pickle
import time
from sklearn.svm import SVR
print("Successfully imported!")


# In[ ]:


print("Reading Datafile...")
updated = pd.read_csv('checkpoint-5.csv')
print("Success!")


# In[ ]:


updated.head(20)


# In[ ]:


print("Obtaining label...")
cols = 'fare_amount'
label = updated[[cols]]
label
print("Obtained labels!")


# In[ ]:


#checkpoint5 = updated.drop(columns = ['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'dist', 'time'])
checkpoint5 = updated.drop(columns = ['fare_amount', 'time'])


# In[ ]:


checkpoint5.head(20)
print("Dropped columns!")


# In[ ]:


# Splitting Dataset into train and test (20% TEST, 80% TRAIN) 
print('Splitting Dataset...')
X_train, X_test, y_train, y_test = train_test_split(checkpoint5, label, test_size = 0.2)
print("Dataset splitted!")


# ### Normalising the values

# In[ ]:


X_sc = StandardScaler()
y_sc = StandardScaler()


# In[ ]:


X_train = X_sc.fit_transform(X_train)
y_train = y_sc.fit_transform(y_train)


# In[ ]:


print('Fitting the model...')
regressor = SVR(kernel = 'rbf')
t0 = time.time()
regressor.fit(X_train, y_train.ravel())
t1 = time.time()
print('Succeeded in fitting model!')


# In[ ]:


print('Predicting...')
y_pred = regressor.predict(X_sc.transform(X_test))
t2 = time.time()
y_pred = y_sc.inverse_transform(y_pred)
print('Predicted all!')


# In[ ]:


y_test = y_test.flatten()
df = pd.DataFrame({'Predicted value': y_pred, 'Real Value': y_test})
df


# In[ ]:


time_train = t1-t0
time_predict = t2-t1

#Get Report
print("SVM Accuracy Score -> ",accuracy_score(y_pred,y_test)*100)
# print("Training time: %fs; Prediction time: %fs" % (time_train, time_predict))
report = classification_report(y_test, y_pred, output_dict=True)

saved_report = pd.DataFrame(report).transpose()

saved_report['Mean squared error'] = mean_squared_error(y_test, y_pred)
saved_report['Training Time'] = time_train;
# TP, FP, TN, FN = perf_measure(y_test, y_pred);

# saved_report['True Positive'] = TP;
# saved_report['False Positive'] = FP;
# saved_report['True Negative'] = TN;
# saved_report['False Negative'] = FN;
# saved_report['Training Time'] = time_train;
# saved_report['Prediction Time'] = time_predict;

# print("True Positive: ", TP)
# print("False Positive: ", FP)
# print("True Negative: ", TN)
# print("False Negative: ", FN)

# Save Report
print("Saving report...")
saved_report.to_csv("SVM_report.csv", index=False)
print("Saved report! ")


# In[ ]:


print("Saving model...")

with open('SVM_Experiment1.pkl', 'wb') as f:

   pickle.dump(regressor, f)

print("Saved model!")


# In[ ]:


# with open('SVM_Experiment1.pkl', 'rb') as f:

#    regressor = pickle.load(f)


# In[ ]:





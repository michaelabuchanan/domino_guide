#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import sys


# In[ ]:


lr = sys.argv[1]


# In[6]:


data = pd.read_csv('/domino/datasets/local/Titanic/processed_train.csv')


# In[7]:


y = data.pop('Survived')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[8]:


model = xgb.XGBClassifier(use_label_encoder=False, learning_rate=lr)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[1]:


with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"Learning Rate": lr, "Accuracy": accuracy}))


# In[ ]:





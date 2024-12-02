#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


dataset = pd.read_csv(r'C:\Users\user\Downloads\Titanic.csv')


# In[3]:


dataset


# In[5]:


dataset.head(5)


# In[7]:


dataset.tail(5)


# In[8]:


dataset.describe()


# In[9]:


dataset.info()


# In[10]:


dataset.columns


# In[11]:


dataset.count()


# In[14]:


l = LabelEncoder()
for i in dataset.select_dtypes(include='object').columns:
    dataset[i] = le.fit_transform(dataset[i])


# In[16]:


x = dataset['PassengerId']
y = dataset['Survived']


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[18]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[19]:


from sklearn.svm import SVC

x_train_reshaped = x_train.values.reshape(-1, 1)
x_test_reshaped = x_test.values.reshape(-1, 1)

model_svc = SVC()
model_svc.fit(x_train_reshaped, y_train)

model_predictions = model_svc.predict(x_test_reshaped)

print(model_predictions)


# In[20]:


model_accuracy_score = accuracy_score(y_test, model_predictions)

print("-- Model Accuracy Score: ", end='')
print(round(model_accuracy_score,3))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("C:/Users/vinay/Downloads/bank-full.csv")
data.head()


# In[5]:


data.drop(["education","contact","month"],inplace=True,axis = 1)


# In[6]:


data


# In[7]:


data2=pd.get_dummies(data,columns=['job','default','marital','housing','loan','poutcome','Y']);data2


# In[8]:


data2.isnull().sum()


# In[80]:


X = data2.iloc[:,0:32]
Y = data2.iloc[:,32] 


# In[81]:


X


# In[82]:


classifier = LogisticRegression()
classifier.fit(X,Y)


# In[83]:


y_pred = classifier.predict(X);y_pred


# In[84]:


y_pred_df= pd.DataFrame({'actual': Y,
                         'predicted_prob': y_pred})


# In[41]:


y_pred_df


# In[42]:


data2['y_pred'] = y_pred


# In[43]:


data2


# In[44]:


#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)


# In[45]:


#accuracy
((1101+39165)/(1101+757+4188+39165))*100


# In[ ]:





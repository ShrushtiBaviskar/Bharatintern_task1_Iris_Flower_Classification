#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


# In[6]:


#Import iris dataset
df=pd.read_csv(r'C:\Users\Shravi\Desktop\irisFlower\iris.csv')


# df

# In[7]:


df.info()


# In[8]:


#checking for null values
df.isnull().sum()


# In[9]:


df.columns


# In[10]:


#Drop unwanted columns
df=df.drop(columns="Id")


# In[11]:


df


# In[16]:


df['Species'].value_counts()


# In[17]:


x=df.iloc[:,:4]
y=df.iloc[:,4]


# In[18]:


x


# In[19]:


y


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[24]:


x_train.shape

x_test.shape

y_train.shape


y_test.shape


# In[25]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[26]:


model.fit(x_train,y_train)


# In[27]:


y_pred=model.predict(x_test)


# In[30]:


y_pred


# In[31]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[32]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("fake_news.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data=data.drop(['id'], axis=1)


# In[7]:


data=data.fillna('')


# In[8]:


data['content']=data['author']+' '+ data['title']+' '+data['text']


# In[9]:


data=data.drop(['title','author','text'], axis=1)


# In[10]:


data.head()


# In[11]:


data['content']=data['content'].apply(lambda x: "".join(x.lower() for x in x.split()))


# In[12]:


data['content']=data['content'].str.replace('[^\w\s]','')


# In[13]:


import nltk
nltk.download('stopwords')


# In[14]:


from nltk.corpus import stopwords
stop=stopwords.words('english')
data['content']=data['content'].apply(lambda x: "".join(x for x in x.split() if x not in stop))


# In[15]:


get_ipython().system('pip install textblob')


# In[18]:


from nltk.stem import WordNetLemmatizer
from textblob import Word
data['content']=data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['content'].head()


# In[19]:


X=data[['content']]
y=data['label']


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=45, stratify=y)


# In[23]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[ ]:





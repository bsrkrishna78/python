#!/usr/bin/env python
# coding: utf-8

# In[2]:


from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[6]:


df=pd.read_csv(r"C:\Users\bsrkr\Downloads\covid_fake.csv")


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df['label'].value_counts()


# In[10]:


df.loc[5:15]


# In[ ]:





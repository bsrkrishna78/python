#!/usr/bin/env python
# coding: utf-8

# In[1]:


Text="Iam learning NLP"


# In[2]:


import pandas as pd
pd.get_dummies(Text.split())


# In[4]:


text=["i love NLP and i will learn NLP in 2month"]


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(text)
vector = vectorizer.transform(text)


# In[8]:


print(vectorizer.vocabulary_)
print(vector.toarray())


# In[11]:


df = pd.DataFrame(
    data=vector.toarray(),
    columns=vectorizer.get_feature_names_out())
df


# In[12]:


text='Iam learning NLP'


# In[13]:


from textblob import TextBlob
TextBlob(text).ngrams(1)


# In[16]:


#For Bigram : For bigrams, use n=2
TextBlob(text).ngrams(2)


# In[17]:


TextBlob(text).ngrams(3)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

reviews_datasets = pd.read_csv(r"C:\Users\bsrkr\Downloads\Reviews.csv")
reviews_datasets = reviews_datasets.head(2000)
reviews_datasets.dropna()


# In[5]:


reviews_datasets['Text'][350]


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix=count_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))


# In[20]:


get_ipython().run_line_magic('pinfo', 'CountVectorizer')


# In[21]:


doc_term_matrix


# In[22]:


print(doc_term_matrix)


# In[23]:


from sklearn.decomposition import LatentDirichletAllocation
LDA=LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)


# In[28]:


import random
for i in range(10):
    random_id=random.randint(0,len(count_vect.get_feature_names_out()))
    print(count_vect.get_feature_names_out()[random_id])


# In[29]:


first_topic=LDA.components_[0]


# In[30]:


top_topic_words=first_topic.argsort()[-10:]


# In[31]:


for i in top_topic_words:
    print(count_vect.get_feature_names_out()[i])


# In[34]:


for i, topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names_out() [i] for i in topic.argsort()[-10:]])
    print('\n')


# In[ ]:





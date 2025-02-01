#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import libraries
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

# Lower casing and removing punctuations
df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Text'] = df['Text'].str.replace('[^\w\s]', ' ')

# Removal of stop words
stop = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Spelling correction
df['Text'] = df['Text'].apply(lambda x: str(TextBlob(x).correct()))
# Lemmatization
df['Text'] = df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df.Text.head()


# In[3]:


get_ipython().system('pip install textblob')


# In[5]:


import pandas as pd

# Example of creating a DataFrame
data = {'Text': ['This is a sample sentence.', 'Another example here!']}
df = pd.DataFrame(data)

# Now you can apply the transformations
df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Text'] = df['Text'].str.replace('[^\w\s]', ' ')

# If you want to see the output
print(df)


# In[6]:


# Import necessary libraries
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
import pandas as pd
# Lower casing and removing punctuations
df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Text'] = df['Text'].str.replace('[^\w\s]', ' ', regex=True)

# Removal of stop words
stop = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Spelling correction
df['Text'] = df['Text'].apply(lambda x: str(TextBlob(x).correct()))

# Lemmatization
df['Text'] = df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Display the first few rows of the processed text
print(df['Text'].head())


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
reviews = df
reviews.dropna(inplace=True)
reviews.Score.hist(bins=5, grid=False)
plt.show()
print(reviews.groupby('Score').count().Id)
reviews['rating'].hist(bins=5, grid=False)


# In[12]:


# Plotting the histogram of the 'Score' column
df['Score'].hist(bins=5, grid=False)
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
print(df.groupby('Score').count())


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Review': [
        'This product is amazing!', 
        'Terrible quality, would not buy again.', 
        'Good value for the price.', 
        'Not bad, but could be better.', 
        'Worst purchase I’ve made in a while.'
    ],
    'Score': [5, 1, 4, 3, 1]
}
df = pd.DataFrame(data)
df.dropna(inplace=True)
df['Score'].hist(bins=5, grid=False)
plt.title('Distribution of Review Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
print(df.groupby('Score').count())


# In[14]:


score_1 = reviews[reviews['Score'] == 1].sample(n=18)
score_2 = reviews[reviews['Score'] == 2].sample(n=18)
score_3 = reviews[reviews['Score'] == 3].sample(n=18)
score_4 = reviews[reviews['Score'] == 4].sample(n=18)
score_5 = reviews[reviews['Score'] == 5].sample(n=18)
reviews_sample = pd.concat([score_1,score_2,score_3,score_4,score_5],axis=0)
reviews_sample.reset_index(drop=True,inplace=True)
print(reviews_sample.groupby('Score').count().Id)


# In[ ]:





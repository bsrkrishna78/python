#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv("train.csv")
train.head()


# In[2]:


train


# In[3]:


train.info()


# In[4]:


train.isna().sum()


# In[5]:


train.describe()


# In[6]:


train.describe(include=[object])


# In[7]:


train['Loan_Status'].value_counts()


# In[9]:


import seaborn as sns
sns.countplot(x=train['Loan_Status'])


# In[10]:


train['Gender'].value_counts()
sns.countplot(x=train['Gender'])


# In[12]:


sns.countplot(x=train['Gender'],hue=train['Loan_Status'])


# In[13]:


sns.countplot(x='Married',data=train,hue='Loan_Status')


# In[14]:


sns.displot(train['LoanAmount'],kde=True)


# In[15]:


import matplotlib.pyplot as plt
grid=sns.FacetGrid(train,row='Gender',col='Married',height=3.2,aspect=1.6)
grid.map(plt.hist,'ApplicantIncome',alpha=.5,bins=10)
grid.add_legend()


# In[16]:


grid.add_legend()


# In[17]:


sns.pairplot(train,hue='Loan_Status',height=2.5)


# In[18]:


train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])


# In[19]:


train['Married'] = train['Married'].fillna(train['Married'].mode()[0])


# In[20]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


# In[21]:


train.head(3)


# In[23]:


train['total_income'] = train['ApplicantIncome'] + train['CoapplicantIncome']


# In[ ]:


rel_feat =['Gender','Married','Dependents','Education','Self_Emmployed','LoanAmount',]


#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split as tts


# In[11]:


data = pd.read_csv(r'train.csv')
print (data)


# In[12]:


data.head()


# data.shape

# In[13]:


data.shape


# In[14]:


data.columns


# In[15]:


data['sentiment'].value_counts()


# In[16]:


data.head(10)


# In[17]:


data.tail(5)


# In[18]:


def clean_text1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    return text

cleaned1=lambda x:clean_text1(x)


# In[ ]:


data['review']=pd.DataFrame(data.review.apply(cleaned2))


# In[24]:


data.head()


# In[25]:


x = data.iloc[0:,0].values
y = data.iloc[0:,1].values


# In[26]:


xtrain,xtest,ytrain,ytest = tts(x,y,test_size = 0.25,random_state = 225)


# In[27]:



tf = TfidfVectorizer()
from sklearn.pipeline import Pipeline


# In[28]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
model=Pipeline([('vectorizer',tf),('classifier',classifier)])

model.fit(xtrain,ytrain)


# In[31]:


ypred=model.predict(xtest)


# In[32]:


accuracy_score(ypred,ytest)


# In[33]:


A=confusion_matrix(ytest,ypred)
print(A)


# In[34]:


recall=A[0][0]/(A[0][0]+A[1][0])
precision=A[0][0]/(A[0][0]+A[0][1])
F1=2*recall*precision/(recall+precision)
print(F1)


# In[ ]:





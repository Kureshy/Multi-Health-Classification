#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


dataset = 'breast-cancer.csv'
ds = pd.read_csv(dataset)
ds


# In[3]:


ds.shape


# In[4]:


print("        ***** RAW DATASET METADATA*****\n\n")
ds.info()


# In[5]:


ds = ds[[cols for cols in ds if ('_mean' in cols) or ('diagnosis' in cols)]]
ds['diagnosis'] = ds['diagnosis'].map({'M':1, 'B':0})
print("                  ***** DATA CLEANED and are READY for FURTHER ANALYSIS ***** ")
ds


# In[6]:


print("                 ***** FEATURES in a STATISTICAL VIEW ***** ")
ds.describe().T


# In[7]:


corr = ds.corr()
corr


# In[8]:


print("            ***** CORRELATION of the FEATURES ***** ")
sns.heatmap(corr, annot=True)


# In[9]:


print("   ***** MAJOR CONTRIBUTORS to the diagnosis ***** ")
ds.corr()['diagnosis'].sort_values().plot(kind='bar')


# In[10]:


print(" ***** COUNTS of 1(MALIGNANT) vs. 0(BENIGN) ***** ")
ds['diagnosis'].value_counts().plot.bar()


# In[11]:


print(" ***** DISTRIBUTION of a SINGLE FEATURE as well as its RELATIONSHIPS with other FEATURES ***** ")
sns.pairplot(ds, hue = 'diagnosis')


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# In[13]:


X = ds.drop('diagnosis', axis=1)
y = ds['diagnosis']


# In[14]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, shuffle = True)


# In[16]:


models = [RandomForestClassifier(), KNeighborsClassifier(), GaussianNB(), SGDClassifier()]


# In[17]:


for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('\n\nModel:', model)
    print('Accuracy:',accuracy_score(y_test, y_pred))
    print('Precision:',precision_score(y_test, y_pred))
    print('Recall:',recall_score(y_test, y_pred))
    print('F1:',f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(sns.heatmap(confusion_matrix(y_test, y_pred), annot=True))


# In[18]:


print("Models' overall scores are 93%-96%!")


# In[ ]:





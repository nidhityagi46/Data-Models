#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('Dataset.data', sep='\t', names=["age","workclass","fnlwgt","education","education-num","marital-status",
                                                  "occupation","relationship","race","sex","capital-gain","capital-loss",
                                                  "hours-per-week","native-country","class"],delimiter=r"\s+")
                                                 
data.head(8)


# In[63]:


data.head(15)


# In[45]:


data.describe(include ="all")

nullvalues = data.apply(lambda x: True if '?' in list(x) else False, axis=1)

len(nullvalues[nullvalues])


# In[46]:


data.drop(nullvalues[nullvalues].index,axis=0,inplace=True)
data.describe()

corr = data.corr()
corr


sns.heatmap(corr)

# Following the regression equation, our dependent variable (y) 
y = data ['class']
# Similarly, our independent variable (x) is the SAT score
x1 = data.iloc[:,0:-1]


# In[49]:


y.head()


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)    
        


# In[50]:


tran_data = MultiColumnLabelEncoder(columns = ['workclass','education','marital-status','occupation','race','relationship','sex','native-country']).fit_transform(x1)


# In[62]:


tran_data.head(10)


# In[52]:


x_train, x_test, y_train, y_test = train_test_split(tran_data.values, y.values, test_size = 0.2,random_state=42)


# In[53]:


from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
svc = SVC()              # The default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)


# In[54]:


prediction = svc.predict(x_test)
prediction


# In[55]:


cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)


# In[56]:


len(x_test)


# In[65]:


d = pd.DataFrame([36,0,212465,9,13,2,0,0,4,1,0,0,40,38])
d = d.values.reshape(1,-1)
svc.predict(d)


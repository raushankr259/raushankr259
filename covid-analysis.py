#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


covid=pd.read_csv("C:\\Users\\DELL\Downloads\\covid.csv")


# In[3]:


covid


# In[4]:


covid.head()


# In[5]:


covid.tail()


# In[6]:


sns.countplot(x='Time',data=covid)


# In[7]:


sns.heatmap(covid.corr())


# In[8]:


T=covid['Deaths'].values
print('Total no of deaths recorded is',sum(T))


# In[9]:


S=covid['Confirmed'].values
print('Total no of cases',sum(S))


# In[10]:


covid.drop(['ConfirmedIndianNational','ConfirmedForeignNational'],axis=1)


# In[13]:


x=covid['Confirmed'].values.reshape(-1,1)
y=covid['Deaths'].values.reshape(-1,1)
y
                            


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7)
regr=LinearRegression()
regr.fit(x_train,y_train)
y_pred=regr.predict(x_test)
print(regr.score(x_test,y_test))
plt.scatter(x_test,y_test,color='r')
plt.plot(x_test,y_pred,color='g')


# In[15]:


xm=x.mean()
ym=y.mean()
print(xm,ym)


# In[16]:


num=0
den=0
s=len(x)
for i in range(s):
    num+=(x[i]-xm)*(y[i]-ym)
    den+=(x[i]-xm)**2
slope=num/den
print('slope of best fit line is',slope)


# In[17]:


c=ym-slope*xm
print('intersept of bfl is',c)


# In[18]:


Y=slope*x+c
plt.plot(x,Y)


# In[19]:


n=int(input('enter the total no of cases'))
deaths=slope*n+c
print('predicted no of death is',deaths)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# ### BarChart

# In[2]:


x=[5,8,10,2,6,12]
y=[12,16,6,10,14,6]
plt.bar(x,y)
plt.title("Bar graph")
plt.ylabel("y axis")
plt.xlabel("x axis")
plt.show()


# ### histogram

# In[3]:


a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a,bins=[0,20,40,60,80,100])
plt.title("histogram")
plt.show()


# In[6]:


import pandas as pd
df=pd.read_csv("student_marks.csv")
df


# In[7]:


plt.hist(df['English'],bins=10)
plt.title('English Histogram')
plt.show()


# In[8]:


plt.boxplot(df['Chemistry'])


# In[11]:


x=df['Maths']
y=df['English']
plt.scatter(x,y)
plt.xlabel('Maths')
plt.ylabel('English')
plt.show()


# In[13]:


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[14]:


dsiris=pd.read_csv('iris.csv')
dsiris
sns.set(style="whitegrid")
ax=sns.stripplot(x='class',y='sepal length',data=dsiris)
plt.title('graph')
plt.show


# In[17]:


iris = pd.read_csv("IRISFLOWER.csv")
iris


# In[20]:


iris.corr()


# In[21]:


sns.heatmap(iris.corr(),annot = True)
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(y, labels = mylabels)
plt.show()


# In[ ]:





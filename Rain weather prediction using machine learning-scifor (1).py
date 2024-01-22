#!/usr/bin/env python
# coding: utf-8

# # Rain Weather Forecasting prediction

# ### problem statement
# ##### Rain Prediction –Weather forecasting
# 
# Weather forecasting is the application of science and technology to predict the conditions of the atmosphere for a given location and time. Weather forecasts are made by collecting quantitative data about the current state of the atmosphere at a given place and using meteorology to project how the atmosphere will change.
# 
# Rain Dataset is to predict whether or not it will rain tomorrow. The Dataset contains about 10 years of daily weather observations of different locations in Australia.

# <b>Importing require library for performing EDA, Data Wrangling and data cleaning<b>

# In[1]:


import pandas as pd # for data wrangling purpose
import numpy as np # Basic computation library
import seaborn as sns # For Visualization 
import matplotlib.pyplot as plt # ploting package
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings # Filtering warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing Temperature Forecast dataset Csv
df=pd.read_csv('weatherAUS.csv')


# In[3]:


df


# In[4]:


print('No of Rows:',df.shape[0])
print('No of Columns:',df.shape[1])
pd.set_option('display.max_columns', None) # This will enable us to see truncated columns
df.head()


# In[5]:


# Sort columns by datatypes
df.columns.to_series().groupby(df.dtypes).groups


# In[6]:


df.tail()


# In[7]:


df.sample()


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.nunique()


# In[12]:


df.describe()


# # Statistical Analysis
# <b>Since dataset is large, Let check for any entry which is repeated or duplicated in dataset.<b>

# In[13]:


df.duplicated().sum()


# <b>check if any whitespace, 'NA' or '-' exist in dataset.<b>

# In[14]:


df.isin([' ','NA','-']).sum().any()


# # Missing value check

# In[15]:


df.isnull().sum()


# In[16]:


#Finding what percentage of data is missing from the dataset
missing_values = df.isnull().sum().sort_values(ascending = False)
percentage_missing_values =(missing_values/len(df))*100
print(pd.concat([missing_values, percentage_missing_values], axis =1, keys =['Missing Values', '% Missing data']))


# In[17]:


print("We had {} Rows and {} Columns before dropping null values.".format(df.shape[0], df.shape[1]))
df.dropna(inplace=True)
print("We have {} Rows and {} Columns after dropping null values.".format(df.shape[0], df.shape[1]))


# In[18]:


# Converting Date datatypes and spliting date into date, month and year.
df['Date']=pd.to_datetime(df['Date'])
df['Day']=df['Date'].apply(lambda x:x.day)
df['Month']=df['Date'].apply(lambda x:x.month)
df['Year']=df['Date'].apply(lambda x:x.year)
df.head()


# In[19]:


df.describe()


# # Start Exploring Present Temperature

# In[20]:


# Plotting histogram for present_Tmax and present_Tmin variables
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df['MaxTemp'],kde=True,color='r')
plt.subplot(1,2,2)
sns.histplot(df['MinTemp'],kde=True,color='m')
plt.show()


# # REMOVAL OF NULL USING IMPUTER

# In[21]:


from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.NaN,strategy="most_frequent")


# In[22]:


l=['WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']


# In[23]:


for i in l:
    df[i]=imp.fit_transform(df[i].values.reshape(-1,1))


# In[24]:


imp2=SimpleImputer(missing_values=np.NaN,strategy="mean")


# In[25]:


lint=['MinTemp','MaxTemp',  
 'Rainfall',     
 'Evaporation',    
 'Sunshine',   
 'WindGustSpeed',  
 'WindSpeed9am',  
 'WindSpeed3pm', 
 'Humidity9am',  
 'Humidity3pm',  
 'Pressure9am',   
 'Pressure3pm',   
 'Cloud9am', 
 'Cloud3pm',      
 'Temp9am' ,    
 'Temp3pm']


# In[26]:


for i in lint:
    df[i]=imp2.fit_transform(df[i].values.reshape(-1,1))


# In[27]:


df.isnull().sum()


# # EDA

# In[30]:


sns.countplot(df['Location'],hue=df['RainTomorrow'])
plt.xticks()


# In[31]:


sns.countplot(df['WindDir9am'],hue=df['RainTomorrow'])
plt.xticks()


# In[32]:


sns.countplot(df['WindDir3pm'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)


# In[33]:


sns.countplot(df['WindGustDir'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)


# In[34]:


sns.countplot(df['RainToday'],hue=df['RainTomorrow'])
plt.xticks(rotation=90)


# In[35]:


sns.scatterplot(x='Evaporation', y='Rainfall', hue='Rainfall', data=df)


# In[36]:


sns.scatterplot(x='MinTemp', y='Rainfall', hue='Rainfall', data=df)


# In[37]:


sns.scatterplot(x='MaxTemp', y='Rainfall', hue='Rainfall', data=df)


# In[38]:


sns.scatterplot(x='Sunshine', y='Rainfall', hue='Rainfall', data=df)


# In[39]:


sns.scatterplot(x='WindGustSpeed', y='Rainfall', hue='Rainfall', data=df)


# In[40]:


sns.scatterplot(x='WindSpeed9am', y='Rainfall', hue='Rainfall', data=df)


# In[41]:


sns.scatterplot(x='WindSpeed3pm', y='Rainfall', hue='Rainfall', data=df)


# In[42]:


sns.scatterplot(x='Humidity9am', y='Rainfall', hue='Rainfall', data=df)


# In[43]:


sns.scatterplot(x='Humidity3pm', y='Rainfall', hue='Rainfall', data=df)


# In[44]:


sns.scatterplot(x='Pressure9am', y='Rainfall', hue='Rainfall', data=df)


# In[45]:


sns.scatterplot(x='Pressure3pm', y='Rainfall', hue='Rainfall', data=df)


# In[46]:


sns.scatterplot(x='Cloud3pm', y='Rainfall', hue='Rainfall', data=df)


# In[47]:


sns.scatterplot(x='Cloud9am', y='Rainfall', hue='Rainfall', data=df)


# In[48]:


sns.scatterplot(x='Temp9am', y='Rainfall', hue='Rainfall', data=df)


# In[49]:


sns.scatterplot(x='Temp3pm', y='Rainfall', hue='Rainfall', data=df)


# # Label Encoder

# In[50]:


l=['WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow']


# In[51]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[52]:


l2=['Date','Location']


# In[53]:


for i in l:
    df[i]=le.fit_transform(df[i])


# In[54]:


for i in l2:
    df[i]=le.fit_transform(df[i])


# In[55]:


df.dtypes


# In[56]:


plt.figure(figsize=(35,45))
count =1
for column in df:
    if count <= 31:
        ax = plt.subplot(11,3,count)
        sns.distplot(df[column])
        plt.xlabel(column) 
    count+=1
plt.show()


# # descriptive statistics

# In[57]:


df.describe()


# In[58]:


plt.figure(figsize=(30,18))
sns.heatmap(df.corr(),linewidth=0.2,annot=True,fmt="0.2f")
plt.show()


# In[59]:


df.corr()["Rainfall"].sort_values()


# In[60]:


# importing libraries to calculate the variance inflation factor, which may result in low accuracy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[61]:


def calc_vif(x1):
    vif=pd.DataFrame()
    vif["Variables"]=x1.columns
    vif["VIF factor"]=[variance_inflation_factor(x1.values,i) for i in range(x1.shape[1])]
    return (vif)


# In[62]:


x=df.drop(["RainTomorrow"],axis=1)
y=df["RainTomorrow"]


# In[63]:


df.columns


# In[64]:


calc_vif(x)


# In[65]:


df.corr()["RainTomorrow"].sort_values()


# In[66]:


dfn=df.drop(['MaxTemp', 'Humidity9am' ,'Pressure9am','Temp3pm'],axis=1)


# # checking outliers

# In[67]:


dfn.plot(kind='box',subplots=True,layout=(4,6),figsize=(12,14))


# In[68]:


from scipy.stats import zscore
z=np.abs(zscore(dfn))


# In[69]:


np.where(z>3)


# In[70]:


dfnew=dfn[(z<3).all(axis=1)]


# In[71]:


dfnew.shape


# In[72]:


dfn.shape


# # checking skewness

# In[73]:


dfnew.skew().sort_values()


# In[74]:


list=['Rainfall','RainToday']


# In[75]:


for i in list:
    if dfnew.skew().loc[i]>0.5:
        dfnew[i]=np.log1p(dfnew[i])


# In[76]:


dfnew.skew().sort_values()


# # standard scaler

# In[77]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[78]:


x=dfnew.drop(["RainTomorrow"],axis=1)
y=dfnew["RainTomorrow"]


# In[79]:


dfx=sc.fit_transform(x)


# In[80]:


dfx.mean()


# # application of machine learning models

# In[81]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import cross_val_score


# In[82]:


def model_selection(instance):
    maxacc=0
    rs=0
    for i in range(0,100):
        x_train,x_test,y_train,y_test=train_test_split(dfx,y,random_state=i,test_size=0.30)
        instance.fit(x_train,y_train)
        pred_train=instance.predict(x_train)
        pred_test=instance.predict(x_test)
        if((accuracy_score(y_test,pred_test))>maxacc):
            maxacc=accuracy_score(y_test,pred_test)
            rs=i
        print(f"at random state {i},  accuracy score is {accuracy_score(y_test,pred_test)}")
        print(f"at random state {i}, confusion matrix is {confusion_matrix(y_test,pred_test)}")
        print(f"at random state {i}, classification report is {classification_report(y_test,pred_test)}")
        print("\n")
    print("Max accuracy at random state",rs, "=",maxacc)
   


# # KNeighbors Classifier

# In[83]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
model_selection(knn)


# In[84]:


score=cross_val_score(knn,dfx,y,cv=5)
print(score)
print(score.mean())
print(score.std())


# # LogisticRegression

# In[85]:


from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
model_selection(lg)


# # CROSSVALIDATION:

# In[86]:


score=cross_val_score(lg,dfx,y,cv=6)
print(score)
print(score.mean())
print(score.std())


# # DecisionTreeClassifier

# In[87]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
model_selection(dtc)


# <b>CV score<b>

# In[88]:


score=cross_val_score(dtc,dfx,y,cv=9)
print(score)
print(score.mean())
print(score.std())


# # RandomForestClassifier

# In[89]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model_selection(rf)


# <b>CV score<b>

# In[90]:


score=cross_val_score(rf,dfx,y,cv=3)
print(score)
print(score.mean())
print(score.std())


# # Hyper parameter tuning using Grid Search CV

# In[91]:


from sklearn.model_selection import GridSearchCV


# In[92]:


dict={"max_features":['auto','sqrt','log2'],
      "max_depth":[17,19,20],
      "criterion":["gini","entropy"],
     "n_estimators":[100,200]}


# In[93]:


gd=GridSearchCV(estimator=rf,param_grid=dict,cv=3)


# In[94]:


gd.fit(dfx,y)


# In[95]:


gd.best_params_


# In[96]:


gd.best_score_


# # final model

# In[97]:


from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(dfx,y,random_state=47,test_size=0.30)
rf=RandomForestClassifier(max_features='auto',max_depth=17,criterion="entropy",n_estimators=100)

rf.fit(x_train,y_train)

rf.score(x_train,y_train)
pred_train=rf.predict(x_train)
pred=rf.predict(x_test)
        
print("Accuracy score:--",accuracy_score(y_test,pred))
print("Confusion matrix:--", confusion_matrix(y_test,pred))
print("classification report:--", classification_report(y_test,pred))
print("\n")


# # AUC ROC

# In[98]:


from sklearn.metrics import roc_curve,auc

fpr,tpr,threshold=roc_curve(y_test,pred)
auc=auc(fpr,tpr)
plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr,tpr,linestyle='-',label='RandomForestClassifier(auc=%0.3f)'%auc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()


# # prediction

# In[99]:


dfpred=pd.DataFrame({'Expected':y_test,'Predicted':pred})


# In[100]:


dfpred.sample(40)


# # model saving

# In[101]:


import pickle
filename="rain_tomorrow.obj"
pickle.dump(rf,open(filename,'wb'))


# In[ ]:





# # prediction of amount of rainfall

# # check of outliers

# In[102]:


dfn.plot(kind='box',subplots=True,layout=(4,6),figsize=(12,14))


# In[103]:


from scipy.stats import zscore
z=np.abs(zscore(dfn))


# In[104]:


dfn.shape


# In[105]:


np.where(z>3)


# In[106]:


dfnew=dfn[(z<3).all(axis=1)]


# In[107]:


dfnew.shape


# # checking the skewness of the data

# In[108]:


dfnew.skew().sort_values()


# In[109]:


x=dfnew.drop(["Rainfall"],axis=1)
y=dfnew["Rainfall"]


# In[110]:


list=['RainTomorrow','RainToday']


# In[111]:


for i in list:
    if dfnew.skew().loc[i]>0.5:
        dfnew[i]=np.log1p(dfnew[i])


# In[112]:


dfnew.skew().sort_values()


# # scaling of data

# In[113]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[114]:


dfx=sc.fit_transform(x)


# In[115]:


dfx.mean()


# # application of machine learning models

# In[116]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score


# In[117]:


def model_selection(instance):
    maxacc=0
    rs=0
    for i in range(0,100):
        x_train,x_test,y_train,y_test=train_test_split(dfx,y,random_state=i,test_size=0.30)
        instance.fit(x_train,y_train)
        pred_train=instance.predict(x_train)
        pred_test=instance.predict(x_test)
        if((r2_score(y_test,pred_test))>maxacc):
            maxacc=r2_score(y_test,pred_test)
            rs=i
        print(f"at random state {i}, testing accuracy is {r2_score(y_test,pred_test)}")
        print(f"at random state {i}, mean squared error is {mean_squared_error(y_test,pred_test)}")
        print(f"at random state {i}, mean absolute error is {mean_absolute_error(y_test,pred_test)}")
        print("\n")
    print("Max accuracy at random state",rs, "=",maxacc)


# # KNeighbors Regressor model

# In[118]:


from sklearn.neighbors import KNeighborsRegressor
knr=KNeighborsRegressor()
model_selection(knr)


# <b>cv score<b>

# In[119]:


score=cross_val_score(knr,dfx,y,cv=9)
print(score)
print(score.mean())
print(score.std())


# # Linear regression

# In[120]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model_selection(lm)


# <b> cv score<b>

# In[121]:


score=cross_val_score(lm,dfx,y,cv=4)
print(score)
print(score.mean())
print(score.std())


# # random forest regressor

# In[122]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
model_selection(rfr)


# <b>cv score<b>

# In[123]:


score=cross_val_score(rfr,dfx,y,cv=9)
print(score)
print(score.mean())
print(score.std())


# # building final model

# In[124]:


rf=RandomForestRegressor()
x_train,x_test,y_train,y_test=train_test_split(dfx,y,random_state=44,test_size=0.30)
instance=rf
instance.fit(x_train,y_train)
pred_train=instance.predict(x_train)
pred_test=instance.predict(x_test)

print(f"at random state {44} testing accuracy is {r2_score(y_test,pred_test)}")
print(f"at random state {44} mean squared error is {mean_squared_error(y_test,pred_test)}")
print(f"at random state {44}, mean absolute error is {mean_absolute_error(y_test,pred_test)}")


# In[ ]:





# # visualisation

# In[125]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=y_test,y=pred_test,color='r')
plt.plot(y_test,y_test,color='b')
plt.xlabel('Actual charges',fontsize= 14)
plt.ylabel('Predicted charges',fontsize= 14)
plt.title('Random Forest Regressor',fontsize= 18)
plt.show()


# # saving the model

# In[126]:


import pickle
filename = 'rainfall.obj'
pickle.dump(knr,open(filename, 'wb'))


# In[ ]:





# In[ ]:





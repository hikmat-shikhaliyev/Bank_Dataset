#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\bank_data_g18.csv')
data


# In[3]:


data.describe(include='all')


# In[4]:


data=data.drop('Profession', axis=1)


# In[5]:


data=data.drop('CIF_Id', axis=1)


# In[6]:


data.dtypes


# In[7]:


data.isnull().sum()


# In[8]:


data.columns


# In[9]:


data.head(2)


# In[10]:


for i in data[['Income', 'Age', 'Amount', 'Overdue_days']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[11]:


data.head(2)


# In[12]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[['Income', 'Amount', 'Overdue_days']]
vif=pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
vif


# In[13]:


data=data.drop('Age', axis=1)


# In[14]:


data.head()


# In[15]:


data.corr()['Default_Flag']


# In[16]:


data=data.drop('Income', axis=1)


# In[17]:


data=data.drop('Amount', axis=1)


# In[18]:


data=data.drop('Overdue_days', axis=1)


# In[19]:


data.head()


# In[20]:


grouped=data.groupby(['Married/Single', 'Default_Flag'])['Default_Flag'].count().unstack().reset_index()
grouped


# In[21]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped


# In[22]:


grouped['Married/Single_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[23]:


data=data.merge(grouped[['Married/Single', 'Married/Single_woe']], how='left', on='Married/Single')
data


# In[24]:


data=data.drop('Married/Single', axis=1)


# In[25]:


grouped=data.groupby(['House_Ownership', 'Default_Flag'])['Default_Flag'].count().unstack().reset_index()


# In[26]:


grouped


# In[27]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['House_Ownership_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[28]:


data=data.merge(grouped[['House_Ownership', 'House_Ownership_woe']], how='left', on='House_Ownership')
data=data.drop('House_Ownership', axis=1)
data


# In[29]:


grouped=data.groupby(['Car_Ownership', 'Default_Flag'])['Default_Flag'].count().unstack().reset_index()
grouped


# In[30]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Car_Ownership_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[31]:


data=data.merge(grouped[['Car_Ownership', 'Car_Ownership_woe']], how='left', on='Car_Ownership')
data=data.drop('Car_Ownership', axis=1)
data


# In[32]:


grouped=data.groupby(['City', 'Default_Flag'])['Default_Flag'].count().unstack().reset_index()
grouped


# In[33]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['City_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[34]:


data=data.merge(grouped[['City', 'City_woe']], how='left', on='City')
data=data.drop('City', axis=1)
data


# In[35]:


grouped=data.groupby(['Product', 'Default_Flag'])['Default_Flag'].count().unstack().reset_index()
grouped


# In[36]:


grouped['positive prop']=grouped[0]/grouped[0].sum()
grouped['negative prop']=grouped[1]/grouped[1].sum()
grouped['Product_woe']=np.log(grouped['positive prop']/grouped['negative prop'])
grouped


# In[37]:


data=data.merge(grouped[['Product', 'Product_woe']], how='left', on='Product')
data=data.drop('Product', axis=1)
data


# In[38]:


data.columns


# In[39]:


data=data[['Married/Single_woe', 'House_Ownership_woe',
       'Car_Ownership_woe', 'City_woe', 'Product_woe', 'Default_Flag']]
data


# In[40]:


X=data.drop('Default_Flag', axis=1)
y=data['Default_Flag']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


def evaluate(model, X_test, y_test):
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1]
    
    roc_prob=roc_auc_score(y_test, y_prob)
    gini_prob=roc_prob*2-1
    
    confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
    report=classification_report(y_test, y_pred)
    
    print('Gini probability:', gini_prob*100)
    print('Confusion_matrix:', confusion_matrix)
    print('Classification report', report)


# In[43]:


lr=LogisticRegression()
lr.fit(X_train, y_train)


# In[44]:


result=evaluate(lr, X_test, y_test)


# In[45]:


y_prob = lr.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
gini = 2*roc_auc-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


# In[46]:


lr_balanced=LogisticRegression(class_weight='balanced')
lr_balanced.fit(X_train, y_train)


# In[47]:


result_balanced=evaluate(lr_balanced, X_test, y_test)


# In[48]:


y_prob = lr_balanced.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
gini = 2*roc_auc-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Auc = %0.2f)' % roc_auc)
plt.plot(fpr, tpr, label='(Gini = %0.2f)' % gini)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='lower right')
plt.show()


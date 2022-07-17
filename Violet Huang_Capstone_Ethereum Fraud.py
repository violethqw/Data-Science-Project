#!/usr/bin/env python
# coding: utf-8

# # Ethereum Fraud Detection

# **Problem Statement**

# Cryptocurrency - such as Ethereum - holds the promise of offering an alternative currency that is decentralised, transparent and safe. Ethereum in particular is supposed to be nearly impossible to hack, and all transactions are cryptographically secured. But where there are humans, there are bad actors. Fraud that has been prevalent in all forms of financial transactions makes no exception when it comes to cryptocurrency. 
# 
# 

# * The starting hypothesis is that a fraudulent ether transaction may have distinguishing features such as less time between the first and the last transactions, and there may be a correlation between the time of the transaction and whether it is a fraudulent transaction.
# 
# * It is acknowledged that the dataset is probably highly imbalanced, and what exactly is the fraud committed is not clear based on the data
# 

# In[1]:


# import everything

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn import svm


# ## Import Dataset
# 

# In[2]:


df = pd.read_csv('transaction_dataset.csv')
df.head(10)


# In[3]:


df.info 


# In[4]:


print(df.columns)


# 
# * "FLAG" refers to whether this is a fraudulent transaction

# In[5]:


df.describe()


# In[6]:


df.var().astype(int)


# In[7]:


#dropping columns with zero variance

df.drop(['Unnamed: 0','Index','Address','min value sent to contract', 'max val sent to contract', 'avg value sent to contract', 'total ether sent contracts', ' ERC20 uniq sent addr.1', ' ERC20 avg time between sent tnx', ' ERC20 avg time between rec tnx',
       ' ERC20 avg time between rec 2 tnx', ' ERC20 avg time between contract tnx', ' ERC20 min val sent contract', ' ERC20 max val sent contract', ' ERC20 avg val sent contract'], axis=1, inplace=True)
df.shape


# ## Cleaning Data

# In[8]:


df.isna().mean()


# In[9]:


# the next step is to single out the numeric data 
cat_cols = []
num_cols= []
for c in df.columns:
    if df[c].map(type).eq(str).any(): #check for strings in column
        cat_cols.append(c)
    else:
        num_cols.append(c)
        
data_num= df[num_cols]
data_cat=df[cat_cols]
data_num.head(10)


# In[10]:


data_num.tail(10)


# In[11]:


# and then fill in the missing value

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data_num)
data_num =  pd.DataFrame(imputer.transform(data_num),columns=num_cols)
data_num.tail(10)


# In[12]:


# Check for Data Duplication
duplicateRowsDf = df[df.duplicated()]
print("Duplicated Entries")
display(duplicateRowsDf)

# Remove Duplicated entries
df2 = df[~df.duplicated()]
df2


# In[13]:


#fill in the missing values


df2 = df2.fillna(method='ffill')

df2


# In[14]:


df2.info()


# # Exploring the Dataset

# In[15]:


(df2['FLAG']).plot(kind='hist')


# In[16]:


d = sns.countplot(df2['FLAG'])
d.set_xticklabels(['Not Fraud','Fraud'])
plt.show()


# * Dataset is imbalanced because majority of the transactions are **not fraud** 
# * accuracy score can therefore be misleading - the model may seem highly accurate because majority of the transactions are not fraud

# In[17]:


df2.groupby(["FLAG"]).size().plot(kind='pie',figsize=(6,8))


# In[18]:


df2.groupby(["Number of Created Contracts"]).size().plot(kind='pie',figsize=(6,8))


# In[19]:


df2.groupby(by = "FLAG").mean().plot(y="Number of Created Contracts", kind = "bar")


# In[20]:


df2.groupby(by = "FLAG").mean().plot(y="Avg min between sent tnx", kind = "bar")


# In[21]:


df2.groupby(by = "FLAG").mean().plot(y="Time Diff between first and last (Mins)", kind="bar")


# * seems like flagged transactions, ie, fraudulent transactions, generally take less amount of time between the first and the last transactions

# In[22]:


df2.groupby(by="FLAG").mean().plot(y="total ether received", kind = "bar")


# In[23]:


#checking if any variables are highly correlated with another
corr=data_num.iloc[:,0:].corr()
print(corr)
pd.set_option('display.max_columns', None)


# In[24]:


corr = df2.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(18,10))
    sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True)


# In[27]:


drop = ['total transactions (including tnx to create contract', ' ERC20 avg val rec',
        ' ERC20 avg val rec',' ERC20 max val rec', ' ERC20 min val rec', ' ERC20 uniq rec contract addr', 'max val sent', ' ERC20 avg val sent',
        ' ERC20 min val sent', ' ERC20 max val sent', ' Total ERC20 tnxs',  'Unique Sent To Addresses',
        'Unique Received From Addresses', 'total ether received', ' ERC20 uniq sent token name', 'min value received', 'min val sent', ' ERC20 uniq rec addr' ]
df2.drop(drop, axis=1, inplace=True)


# In[28]:


df2


# In[31]:


# drop the non-numerical columns
df2.drop([' ERC20 most sent token type',' ERC20_most_rec_token_type'],axis=1)


# In[32]:



corr = df2.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(18,10))
    sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True)


# # Preparing the Data for Modeling

# In[55]:


# Turn object variables into 'category' dtype for more computation efficiency
categories = df2.select_dtypes('O').columns.astype('category')
df2[categories]

df2.drop(df2[categories], axis=1, inplace=True)
df2


# In[56]:


## oversampling using Synthetic minority oversampling technique
import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE


# In[57]:


y = df2.iloc[:, 0]
X = df2.iloc[:, 1:]
print(X.shape, y.shape)


# In[58]:


# Split into training (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ## Oversampling

# In[59]:


# using SMOTE for oversampling because of the imablanced dataset

oversample = SMOTE()
print(f'Shape of the training before SMOTE: {X_train.shape, y_train.shape}')

x_tr_resample, y_tr_resample = oversample.fit_resample(X_train,y_train)
print(f'Shape of the training after SMOTE: {x_tr_resample.shape,y_tr_resample}')


# # Modeling

# In[62]:


from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled)

final_test_scaled = scaler.transform(X_test)
final_test_scaled = pd.DataFrame(final_test_scaled)

X_train_scaled


# In[63]:


print(y_tr_resample)


# ## K nearest neighbours

# In[67]:


#Choosing K nearest neighbours as model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_tr_resample, y_tr_resample)


# In[70]:


y_pred = classifier.predict(X_test)


# In[144]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print (accuracy_score(y_test, y_pred))
print (roc_auc_score(y_test, y_pred))


# ## Logistic Regression

# In[115]:


#Normalisation 

from sklearn.preprocessing import PowerTransformer
norm = PowerTransformer()
norm_train = norm.fit_transform(x_tr_resample)


# In[119]:


LR = LogisticRegression(random_state=42)
LR.fit(norm_train, y_tr_resample)

X_test1 = scaler.transform(X_test)
preds2 = LR.predict(X_test1)
print(preds2)


# In[125]:


print(y_test.shape)
y_test.value_counts()


# In[145]:


from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import plot_confusion_matrix

confusion_matrix(y_test, preds2)

print(cm)
print (accuracy_score(y_test, preds2))
print (roc_auc_score(y_test, preds2))
plot_confusion_matrix(LR, X_test1, y_test)


# Looking at the confusion matrix - 
# 
# LR model, correctly identified 269 (TP) of FRAUD cases, out of 295 actual fraud.
# LR model flagged as FRAUD 522 (FP) out of 1564 NON-FRAUD cases
# 
# 

# Notes
# 
# * the LR model is slightly more sensitive in picking out fraud - sensitivity = TP/TP+FN = 0.9119
# 
# * The false positive rate is on the higher side - FP/FP+TN = 522/1564 = 0.3338;
# * but in the context of fraud detection, sensitivity is arguably more important

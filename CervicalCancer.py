#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
from sklearn import svm

get_ipython().run_line_magic('matplotlib', 'inline')

seed = 42


# In[2]:


# loading data 

def load_cervicalcancer_data():        
    return pd.read_csv("G:\mini project\cervical-cancer-risk-classification\kag_risk_factors_cervical_cancer.csv")

cervical= load_cervicalcancer_data()
len(cervical)


# In[3]:


#replacing ? with 0
cervical=cervical.replace("?", 0)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
cervical.hist( figsize=(15,15))
plt.show()


# In[5]:


cervical.plot.area()


# In[6]:


cervical.info()


# In[7]:


# renaming dx:Cancer to Cancer
cervical = pd.DataFrame(cervical)
cervical.rename(columns = {'Dx:Cancer':'Cancer'}, inplace = True)


# In[8]:


y=cervical["Cancer"]
X=cervical.drop(["Cancer"], axis=1)


# In[9]:


#splitting data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=seed)


# In[10]:


np.sum(y_test.values==1), np.sum(y_test.values==0)


# In[11]:


np.sum(y_train.values==1), np.sum(y_train.values==0)


# In[12]:


cervical.info()


# In[13]:


cervical


# # MODEL 1- using all features

# # 1. RANDOM FOREST

# In[14]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=seed)
rfc.fit(X_train, y_train)


# In[15]:


from sklearn.metrics import accuracy_score
y_pred=rfc.predict(X_test)
accuracy_score(y_pred, y_test)


# In[16]:


from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['Pred_neg','Pred_pos'],index=['neg','pos']))


# In[17]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[18]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
precision_score(y_test, y_pred) 


# In[19]:


recall_score(y_test, y_pred)


# In[20]:


f1_score(y_test, y_pred)


# In[21]:


roc_auc_score(y_test, y_pred)


# In[22]:


skplt.metrics.plot_precision_recall_curve(y_true=y_test, y_probas=rfc.predict_proba(X_test))
plt.show()


# In[23]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
y_scores=rfc.predict_proba(X_test)[:,1]
auc=roc_auc_score(y_test,y_scores)
print(f"AUC score:{auc}")

def plot_roc(fpr,tpr):
    plt.plot(fpr,tpr,color='green',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_scores)
#print(fpr)

plot_roc(fpr,tpr)


# In[24]:


#Evaulating which features the classifier finds important for making its decisions
feats = rfc.feature_importances_

#Create new instance of dataframe
feat_importances=pd.DataFrame()

#set columns in datafram to features and their importances
feat_importances["feature"]=X.columns
feat_importances["rfc"]=feats

#Display data
feat_importances


# # 2. SVC

# In[25]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',gamma=0.1,C=100)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
results=cross_val_score(classifier,X_test,y_test,cv=5,n_jobs=-1)
print(np.mean(results))


# In[26]:


from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['Pred_neg','Pred_pos'],index=['neg','pos']))


# In[27]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score
precision_score(y_test, y_pred) 


# In[28]:


recall_score(y_test, y_pred)


# In[29]:


f1_score(y_test, y_pred)


# In[30]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred)


# In[31]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
classifier=SVC(probability=True)
classifier.fit(X_train,y_train)
y_scores=classifier.predict_proba(X_test)[:,1]
auc=roc_auc_score(y_test,y_scores)
print(f"AUC score:{auc}")

def plot_roc(fpr,tpr):
    plt.plot(fpr,tpr,color='green',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_scores)
#print(fpr)

plot_roc(fpr,tpr)


# In[32]:


import sklearn
import scikitplot as skplt
skplt.metrics.plot_precision_recall_curve(y_true=y_test, y_probas=classifier.predict_proba(X_test))
plt.show()


# # 3. MLP Classifier

# In[33]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

estimator = MLPClassifier(alpha=1e-6, hidden_layer_sizes=(13), max_iter=150, random_state=4, solver='lbfgs')
#model.fit(X,Y)


# In[34]:


from sklearn.metrics import confusion_matrix
estimator.fit(X_train, y_train)
y_pred=estimator.predict(X_test)
results = cross_val_score(estimator, X_test, y_test,cv=10,n_jobs=-1)
print(results.mean())


# In[35]:


print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['pred_neg','pred_pos'],index=['neg','pos']))


# In[36]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score
precision_score(y_test, y_pred) 


# In[37]:


recall_score(y_test, y_pred)


# In[38]:


f1_score(y_test, y_pred)


# In[39]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt

y_scores=estimator.predict_proba(X_test)[:,1]
auc=roc_auc_score(y_test,y_scores)
print(f"AUC score:{auc}")

def plot_roc(fpr,tpr):
    plt.plot(fpr,tpr,color='green',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_scores)
#print(fpr)

plot_roc(fpr,tpr)


# In[40]:


import sklearn
import scikitplot as skplt
skplt.metrics.plot_precision_recall_curve(y_true=y_test, y_probas=estimator.predict_proba(X_test))
plt.figure(figsize=(6, 6))
plt.show()


# # MODEL 2- using top 10 features

# In[41]:


X_train_model2=X_train.drop(['Smokes','Hormonal Contraceptives','IUD'],axis=1)


# In[42]:


X_test_model2=X_test.drop(['Smokes', 'Hormonal Contraceptives','IUD'],axis=1)


# # 1. Random Forest

# In[43]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=seed)
rfc.fit(X_train_model2, y_train)


# In[44]:


from sklearn.metrics import accuracy_score
y_pred=rfc.predict(X_test_model2)
accuracy_score(y_pred, y_test)


# In[45]:


from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['Pred_neg','Pred_pos'],index=['neg','pos']))


# In[46]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
precision_score(y_test, y_pred) 


# In[47]:


f1_score(y_test, y_pred)


# In[48]:


recall_score(y_test, y_pred)


# In[49]:


roc_auc_score(y_test, y_pred)


# In[50]:


skplt.metrics.plot_precision_recall_curve(y_true=y_test, y_probas=rfc.predict_proba(X_test_model2))
plt.show()


# In[51]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
y_scores=rfc.predict_proba(X_test_model2)[:,1]
auc=roc_auc_score(y_test,y_scores)
print(f"AUC score:{auc}")

def plot_roc(fpr,tpr):
    plt.plot(fpr,tpr,color='green',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_scores)
#print(fpr)

plot_roc(fpr,tpr)


# # 2. SVC

# In[52]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',gamma=0.1,C=100)
classifier.fit(X_train_model2, y_train)
y_pred=classifier.predict(X_test_model2)
results=cross_val_score(classifier,X_test_model2,y_test,cv=5,n_jobs=-1)
print(np.mean(results))


# In[53]:


from sklearn.metrics import confusion_matrix
print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['Pred_neg','Pred_pos'],index=['neg','pos']))


# In[54]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score
precision_score(y_test, y_pred) 


# In[55]:


recall_score(y_test, y_pred)


# In[56]:


f1_score(y_test, y_pred)


# In[57]:


roc_auc_score(y_test, y_pred)


# In[58]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
classifier=SVC(probability=True)
classifier.fit(X_train_model2,y_train)
y_scores=classifier.predict_proba(X_test_model2)[:,1]
auc=roc_auc_score(y_test,y_scores)
print(f"AUC score:{auc}")

def plot_roc(fpr,tpr):
    plt.plot(fpr,tpr,color='green',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_scores)
#print(fpr)

plot_roc(fpr,tpr)


# In[59]:


import sklearn
import scikitplot as skplt
skplt.metrics.plot_precision_recall_curve(y_true=y_test, y_probas=classifier.predict_proba(X_test_model2))
plt.show()


# # 3. MLP

# In[60]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

estimator = MLPClassifier(alpha=1e-6, hidden_layer_sizes=(13), max_iter=150, random_state=4, solver='lbfgs')
#model.fit(X,Y)


# In[61]:


from sklearn.metrics import confusion_matrix
estimator.fit(X_train_model2, y_train)
y_pred=estimator.predict(X_test_model2)
results = cross_val_score(estimator, X_test_model2, y_test,cv=10,n_jobs=-1)
print(results.mean())


# In[62]:


print(pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['pred_neg','pred_pos'],index=['neg','pos']))


# In[63]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score
precision_score(y_test, y_pred) 


# In[64]:


roc_auc_score(y_test, y_pred)


# In[65]:


recall_score(y_test, y_pred)


# In[66]:


f1_score(y_test, y_pred)


# In[67]:


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt

y_scores=estimator.predict_proba(X_test_model2)[:,1]
auc=roc_auc_score(y_test,y_scores)
print(f"AUC score:{auc}")

def plot_roc(fpr,tpr):
    plt.plot(fpr,tpr,color='green',label='ROC')
    plt.plot([0,1],[0,1],color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_scores)
#print(fpr)

plot_roc(fpr,tpr)


# In[68]:


import sklearn
import scikitplot as skplt
skplt.metrics.plot_precision_recall_curve(y_true=y_test, y_probas=estimator.predict_proba(X_test_model2))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





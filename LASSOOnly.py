import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from time import time
import sys
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

###################################################
### GSE132903 GSE118553 GSE48350_e.xlsx  GSE5281_e  GSE36980_e
dbName = 'GSE36980_e.xlsx'

print(dbName)
###############################
###Read data and preprocessing
################################
ds = pd.read_excel(dbName)       

ds_t = ds.transpose()
print(np.shape(ds_t))

XX = ds_t.iloc[1:,:-1].values
print('X shape is', XX.shape)
feat = np.array(XX)

label = ds_t.iloc[1:,-1].values
print('Y shape is', label.shape)
label = np.array(label[:], dtype=int)
# ###############################################
scaler = MinMaxScaler(feature_range=(0, 1))
feat = scaler.fit_transform(feat)

print(scaler.data_min_)
print(scaler.data_max_)

t0 = time()
# Create the Lasso regression model
lasso = Lasso(alpha=0.01)  # alpha controls the regularization strength

# Fit the model to the training data
lasso.fit(feat,label)

# Get the coefficients to identify selected features
selected_features = np.where(lasso.coef_ != 0)[0]

# Print the selected features
print("Selected Features:", selected_features)

feat_selected = feat[:, selected_features]

print ('top_features shape is ' , feat_selected.shape)
print ('feat shape is ' , feat.shape)
print ('target_selected  shape is ' , label.shape)

# #############################################
# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat_selected, label, test_size=0.3, stratify=label)

mdl = svm.SVC()

mdl.fit(xtrain, ytrain)
fit_time = time()- t0
print("time %.3f s" %fit_time)
 # accuracy
y_pred    = mdl.predict(xtest)
cm = confusion_matrix(ytest, y_pred) 

accuracy_score=metrics.accuracy_score(ytest, y_pred)
print("accuracy_score=",accuracy_score)

 ################################# f1_score
from sklearn.metrics import f1_score
f1_sco=f1_score(ytest, y_pred)
print('f1_score=',f1_sco)        
  
 ########################### roc_auc_score  
roc_auc = metrics.roc_auc_score(ytest, y_pred)
print("roc_auc=",roc_auc)
             
 ################ precision
PrecisionScore = precision_score(ytest, y_pred)
print(" Precision:",PrecisionScore)
       
 ################Recall       
Rec=recall_score(ytest, y_pred)
print("recall:", Rec)
        
 ###################specificity
spec = cm[1,1]/(cm[1,1]+cm[1,0])
print('Specificity : ', spec)
         
 ############# kappa
kappa=cohen_kappa_score(ytest, y_pred)
print('kappa:', kappa)
         
print(fit_time,"  ",len(selected_features),"  ",accuracy_score,"  ",f1_sco,"  ",
      roc_auc,"   ",PrecisionScore,"   ",Rec,"  ",spec,"   ",kappa)        


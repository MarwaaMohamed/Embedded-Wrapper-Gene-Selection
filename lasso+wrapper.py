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
### GSE132903 GSE118553

ModelName = "HHo"
CFName = "SVM"
# parameter
N    = 10    # number of solutions / population size ( for all methods )
T    = 10   # maximum number of generations

dbName = 'GSE132903_e.xlsx'


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
target_selected = np.array(label[:], dtype=int)

# ###############################################
scaler = MinMaxScaler(feature_range=(0, 1))
feat = scaler.fit_transform(feat)

print(scaler.data_min_)
print(scaler.data_max_)

###########################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feat, label, test_size=0.3, random_state=42)

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
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

k    = 5     # k-value in KNN
####################################################
if ModelName == "HHo" :
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T , 'CFName':CFName}
    from FS.hho import jfs   # change this to switch algorithm 
elif  ModelName == "GWO" :
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T,'CFName':CFName}
    from FS.gwo import jfs 
elif  ModelName == "GA" :
    CR   = 0.8
    MR   = 0.01
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR,'CFName':CFName}
    from FS.ga import jfs 
elif  ModelName == "PSO" :
    w    = 0.9
    c1   = 2
    c2   = 2
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2,'CFName':CFName}
    from FS.pso import jfs 
elif  ModelName == "WOA" :
    b  = 1    # constant
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'b':b,'CFName':CFName}  
    from FS.woa import jfs 
elif  ModelName == "ja" :
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T , 'CFName':CFName}
    from FS.ja import jfs 
elif  ModelName == "ssa" :
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T , 'CFName':CFName}
    from FS.ssa import jfs 
elif  ModelName == "sca" :
    opts = {'k':k, 'fold':fold, 'N':N, 'T':T , 'CFName':CFName}
    from FS.sca import jfs 
#####################################################    
for i in range(1):
    
        print("\n iteration ",i)  
        # perform feature selection
        t0 = time()
        fmdl = jfs(feat_selected, label, opts)
        sf   = fmdl['sf']
        
        # model with selected features
        num_train = np.size(xtrain, 0)
        num_valid = np.size(xtest, 0)
        x_train   = xtrain[:, sf]
        y_train   = ytrain.reshape(num_train)  # Solve bug
        x_valid   = xtest[:, sf]
        y_valid   = ytest.reshape(num_valid)  # Solve bug
        
        ##########################Model
        if CFName == "KNN" :
                  mdl       = KNeighborsClassifier(n_neighbors = k) 
        elif CFName == "SVM" :
                  mdl = svm.SVC() 
        elif  CFName == "MLP" :   
                  mdl = MLPClassifier(hidden_layer_sizes=(5,4), 
                            activation='relu', solver='adam', learning_rate='constant',
                          learning_rate_init=0.01, max_iter=1500,random_state=33)
        ######################################################## 
        mdl.fit(x_train, y_train)
        fit_time = time()- t0
        print("time %.3f s" %fit_time)
        # accuracy
        y_pred    = mdl.predict(x_valid)
        Acc       = np.sum(y_valid == y_pred)  / num_valid
        print("Accuracy:", 100 * Acc)
        
        # # # # # ####### metrics to evaluate classifier
        # # ############ confusion matrix
        
        cm = confusion_matrix(y_valid, y_pred) 
        #print('Confusion Matrix : \n', cm)
               
        ################ Accuracy
        accuracy_score=metrics.accuracy_score(y_valid, y_pred)
        print("accuracy_score=",accuracy_score)
              
        ################################# f1_score
        from sklearn.metrics import f1_score
        f1_sco=f1_score(y_valid, y_pred)
        print('f1_score=',f1_sco)        
         
        ########################### roc_auc_score  
        roc_auc = metrics.roc_auc_score(y_valid,y_pred)
        print("roc_auc=",roc_auc)
                    
        ################ precision
        PrecisionScore = precision_score(y_valid, y_pred)
        print(" Precision:",PrecisionScore)
              
        ################Recall       
        Rec=recall_score(y_valid, y_pred)
        print("recall:", Rec)
               
        ###################specificity
        spec = cm[1,1]/(cm[1,1]+cm[1,0])
        print('Specificity : ', spec)
                
        ############# kappa
        kappa=cohen_kappa_score(y_valid, y_pred)
        print('kappa:', kappa)
                
        # number of selected features
        num_feat = fmdl['nf']
        print("Feature Size:", num_feat)
        
        print(fit_time,"  ",num_feat,"  ",100 * Acc,"  ",f1_sco,"  ",
              roc_auc,"   ",PrecisionScore,"   ",Rec,"  ",spec,"   ",kappa)
        # plot convergence
        curve   = fmdl['c']
        curve   = curve.reshape(np.size(curve,1))
        x       = np.arange(0, opts['T'], 1.0) + 1.0
        
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(x, curve, 'o-')
        ax.set_xlabel('Number of Generations',fontsize=20)
        ax.set_ylabel('Fitness',fontsize=20)
        ax.set_title(ModelName + '  ' + CFName,fontsize=20)
        ax.grid()
        plt.show()
      





import numpy as np
import pandas as pd
import sys
import ast # to convert list to string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
#importing VOROS
import VOROS

def create_models(random_state):
    # load the wisconsin data from sklearn
    X,y = load_breast_cancer(return_X_y=True,as_frame=True)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=random_state, stratify=y)

               
    mod1 = LogisticRegression(random_state=random_state)
    mod2 = MultinomialNB()
    prep=MinMaxScaler()
    X_train_scaled=prep.fit_transform(X_train)
    X_test_scaled=prep.transform(X_test)
    mod3 = RandomForestClassifier()
    
    print("Fitting First Model")
  
    mod1.fit(X_train,y_train)
    print("Predicting on First Model")
    mod1_train_preds=mod1.predict_proba(X_train)[:,1]
    mod1_test_preds=mod1.predict_proba(X_test)[:,1]
    print("Fitting Second Model")
    mod2.fit(X_train_scaled,y_train)
    print("Predicting on Second Model")
    mod2_train_preds=mod2.predict_proba(X_train_scaled)[:,1]
    mod2_test_preds=mod2.predict_proba(X_test_scaled)[:,1]
    print("Fitting Third Model")
    mod3.fit(X_train,y_train)
    print("Predicting on Third Model")
    mod3_train_preds=mod3.predict_proba(X_train)[:,1]
    mod3_test_preds=mod3.predict_proba(X_test)[:,1]
    print("Calculating AUC")
    # Calculate AUROCs
    mod1_train_auroc = roc_auc_score(y_train,mod1_train_preds)
    mod1_test_auroc = roc_auc_score(y_test,mod1_test_preds)
    
    mod2_train_auroc = roc_auc_score(y_train,mod2_train_preds)
    mod2_test_auroc = roc_auc_score(y_test,mod2_test_preds)
    
    mod3_train_auroc = roc_auc_score(y_train,mod3_train_preds)
    mod3_test_auroc = roc_auc_score(y_test,mod3_test_preds)
    
    print("Calculating VOROS")
    # Calculate VOROS
    mod1_train_fpr,mod1_train_tpr,_=roc_curve(y_train, mod1_train_preds)
    mod1_test_fpr,mod1_test_tpr,_=roc_curve(y_test, mod1_test_preds)
    
    mod2_train_fpr,mod2_train_tpr,_=roc_curve(y_train, mod2_train_preds)
    mod2_test_fpr,mod2_test_tpr,_=roc_curve(y_test, mod2_test_preds)
    
    mod3_train_fpr,mod3_train_tpr,_=roc_curve(y_train, mod3_train_preds)
    mod3_test_fpr,mod3_test_tpr,_=roc_curve(y_test, mod3_test_preds)
    
    mod1_train_lst=[[mod1_train_fpr[i],mod1_train_tpr[i]] for i in range(len(mod1_train_fpr))]
    mod1_test_lst=[[mod1_test_fpr[i],mod1_test_tpr[i]] for i in range(len(mod1_test_fpr))]
    
    mod2_train_lst=[[mod2_train_fpr[i],mod2_train_tpr[i]] for i in range(len(mod2_train_fpr))]
    mod2_test_lst=[[mod2_test_fpr[i],mod2_test_tpr[i]] for i in range(len(mod2_test_fpr))]
    
    mod3_train_lst=[[mod3_train_fpr[i],mod3_train_tpr[i]] for i in range(len(mod3_train_fpr))]
    mod3_test_lst=[[mod3_test_fpr[i],mod3_test_tpr[i]] for i in range(len(mod3_test_fpr))]
    
    
    mod1_train_voros = VOROS.Volume(mod1_train_lst,[0,1])
    mod1_test_voros = VOROS.Volume(mod1_test_lst,[0,1])

    
    mod2_train_voros = VOROS.Volume(mod2_train_lst,[0,1])
    mod2_test_voros = VOROS.Volume(mod2_test_lst,[0,1])
    
    mod3_train_voros = VOROS.Volume(mod3_train_lst,[0,1])
    mod3_test_voros = VOROS.Volume(mod3_test_lst,[0,1])
    
    data_dict = {'mod1_train_auroc': mod1_train_auroc,
                 'mod1_train_voros': mod1_train_voros,
                 'mod1_test_auroc' : mod1_test_auroc,
                 'mod1_test_voros' : mod1_test_voros,
                 'mod1_train_curve' : VOROS.cull_pnts(mod1_train_lst),
                 'mod1_test_curve' : VOROS.cull_pnts(mod1_test_lst),
                 
                 'mod2_train_auroc': mod2_train_auroc,
                 'mod2_train_voros': mod2_train_voros,
                 'mod2_test_auroc' : mod2_test_auroc,
                 'mod2_test_voros' : mod2_test_voros,
                 'mod2_train_curve' : VOROS.cull_pnts(mod2_train_lst),
                 'mod2_test_curve' : VOROS.cull_pnts(mod2_test_lst),
                 
                 'mod3_train_auroc': mod3_train_auroc,
                 'mod3_train_voros': mod3_train_voros,
                 'mod3_test_auroc' : mod3_test_auroc,
                 'mod3_test_voros' : mod3_test_voros,
                 'mod3_train_curve' : VOROS.cull_pnts(mod3_train_lst),
                 'mod3_test_curve' : VOROS.cull_pnts(mod3_test_lst),
                }
                 
    return data_dict

if __name__=='__main__':
    state=int(input("Input a random state: "))
    preds=create_models(state)
    log_curve,naive_curve,forest_curve=preds['mod1_test_curve'],preds['mod3_test_curve'],preds['mod2_test_curve']
    print(f"Log AUROC: {preds['mod1_test_auroc']}\nLog VOROS: {preds['mod1_test_VOROS']}\nNaive AUROC: {preds['mod3_test_auroc']}\nNaive VOROS: {preds['mod3_test_VOROS']}\nForest AUROC: {preds['mod2_test_auroc']}\nForest VOROS: {preds['mod2_test_VOROS']}\n")
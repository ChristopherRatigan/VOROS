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
# Importing VOROS
import VOROS

def create_models(file_path, random_state):
    # Load creditcard.csv
    df = pd.read_csv(file_path)
    X1=pd.DataFrame({'data':df.iloc[:,1]})
    X2=pd.DataFrame({'data':df.iloc[:,2]})
    y=df['Class']
    
    X1_train,X1_test,X2_train,X2_test,y_train,y_test = train_test_split(X1,X2,y, random_state=random_state, stratify=y,train_size=0.05)

               
    mod1 = LogisticRegression(C=1, max_iter=50)
    
    mod2 = LogisticRegression(C=1, max_iter=50)
    
    print("Fitting First Model")
  
    mod1.fit(X1_train,y_train)
    print("Predicting on First Model")
    mod1_train_preds=mod1.predict_proba(X1_train)[:,1]
    mod1_test_preds=mod1.predict_proba(X1_test)[:,1]
    print("Fitting Second Model")
    mod2.fit(X2_train,y_train)
    print("Predicting on Second Model")
    mod2_train_preds=mod2.predict_proba(X2_train)[:,1]
    mod2_test_preds=mod2.predict_proba(X2_test)[:,1]
    print("Calculating AUC")
    # Calculate AUROCs
    mod1_train_auroc = roc_auc_score(y_train,mod1_train_preds)
    mod1_test_auroc = roc_auc_score(y_test,mod1_test_preds)
    
    mod2_train_auroc = roc_auc_score(y_train,mod2_train_preds)
    mod2_test_auroc = roc_auc_score(y_test,mod2_test_preds)
    
    print("Calculating VOROS")
    # Calculate VOROS
    mod1_train_fpr,mod1_train_tpr,_=roc_curve(y_train, mod1_train_preds)
    mod1_test_fpr,mod1_test_tpr,_=roc_curve(y_test, mod1_test_preds)
    
    mod2_train_fpr,mod2_train_tpr,_=roc_curve(y_train, mod2_train_preds)
    mod2_test_fpr,mod2_test_tpr,_=roc_curve(y_test, mod2_test_preds)
    
    mod1_train_lst=[[mod1_train_fpr[i],mod1_train_tpr[i]] for i in range(len(mod1_train_fpr))]
    mod1_test_lst=[[mod1_test_fpr[i],mod1_test_tpr[i]] for i in range(len(mod1_test_fpr))]
    
    mod2_train_lst=[[mod2_train_fpr[i],mod2_train_tpr[i]] for i in range(len(mod2_train_fpr))]
    mod2_test_lst=[[mod2_test_fpr[i],mod2_test_tpr[i]] for i in range(len(mod2_test_fpr))]
    
    mod1_train_voros = VOROS.Volume(mod1_train_lst,[0,1])
    mod1_test_voros = VOROS.Volume(mod1_test_lst,[0,1])

    
    mod2_train_voros = VOROS.Volume(mod2_train_lst,[0,1])
    mod2_test_voros = VOROS.Volume(mod2_test_lst,[0,1])
    
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
                }
                 
    return data_dict

if __name__=='__main__':
    state=int(input("Input a random state: "))
    csv_path=input("Path to creditcard.csv file: ")
    preds=create_models(csv_path, state)
    curve1,curve2=preds['mod1_test_curve'],preds['mod2_test_curve']
    print(f"V1 AUROC: {preds['mod1_test_auroc']}\nV1 VOROS: {preds['mod1_test_VOROS']}\nV2 AUROC: {preds['mod2_test_auroc']}\nV2 VOROS: {preds['mod2_test_VOROS']}")
import os
import numpy as np
import pandas as pd
import torch
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

def load_dataset(df):
    X = torch.vstack([torch.load(path).float() for path in df['path'].to_list()]).detach().numpy()
    y = torch.tensor(df.label.to_list()).detach().numpy()
    return X, y

def create_models(labels_path, random_state):
    # Load labels.csv created by our script
    df = pd.read_csv(os.path.join(labels_path, 'labels.csv'), index_col='study_id')
    df['label'] = df['label'].apply(lambda string: ast.literal_eval(string))
    
    df['default_label']=df.groupby("study_id")['label'].agg(lambda x: x.value_counts().idxmax()[2])
    
    train_df,test_df=train_test_split(df, train_size=0.8,random_state=random_state)
        
    # Load data
    X_train, y_train = load_dataset(train_df)
    y_train=y_train[:,2]
    X_test, y_test = load_dataset(test_df)
    y_test=y_test[:,2]

    #scale data for Bayes
    X_train_scaled = MinMaxScaler().fit_transform(X_train)
    X_test_scaled = MinMaxScaler().fit_transform(X_test)
    
    #Instantiate models
    model1=LogisticRegression(C=1000,max_iter=25,random_state=random_state)               
    model2 = RandomForestClassifier(random_state=random_state)
    model3 = MultinomialNB()
    
    
    #fit and calculate ROC curves
    model1.fit(X_train,y_train)
    log_train_preds = model1.predict_proba(X_train)[:,1]
    log_test_preds = model1.predict_proba(X_test)[:,1]
    
    model2.fit(X_train,y_train)
    forest_train_preds = model2.predict_proba(X_train)[:,1]
    forest_test_preds = model2.predict_proba(X_test)[:,1]
    
    model3.fit(X_train_scaled,y_train)
    naive_train_preds = model3.predict_proba(X_train)[:,1]
    naive_test_preds = model3.predict_proba(X_test)[:,1]
    
    # Calculate AUROCs
    log_train_auroc = roc_auc_score(y_train, log_train_preds)
    log_test_auroc = roc_auc_score(y_test, log_test_preds)
    
    forest_train_auroc = roc_auc_score(y_train, forest_train_preds)
    forest_test_auroc = roc_auc_score(y_test, forest_test_preds)
    
    naive_train_auroc = roc_auc_score(y_train, naive_train_preds)
    naive_test_auroc = roc_auc_score(y_test, naive_test_preds)
    
    # Calculate VOROS
    log_train_fpr,log_train_tpr,_=roc_curve(y_train, log_train_preds)
    log_test_fpr,log_test_tpr,_=roc_curve(y_test, log_test_preds)
    
    forest_train_fpr,forest_train_tpr,_ = roc_curve(y_train, forest_train_preds)
    forest_test_fpr,forest_test_tpr,_ = roc_curve(y_test, forest_test_preds)
    
    naive_train_fpr,naive_train_tpr,_ = roc_curve(y_train, naive_train_preds)
    naive_test_fpr,naive_test_tpr,_ = roc_curve(y_test, naive_test_preds)
    
    log_train_lst=[[log_train_fpr[i],log_train_tpr[i]] for i in range(len(log_train_fpr))]
    log_test_lst=[[log_test_fpr[i],log_test_tpr[i]] for i in range(len(log_test_fpr))]
    
    forest_train_lst=[[forest_train_fpr[i],forest_train_tpr[i]] for i in range(len(forest_train_fpr))]
    forest_test_lst=[[forest_test_fpr[i],forest_test_tpr[i]] for i in range(len(forest_test_fpr))]
    
    naive_train_lst=[[naive_train_fpr[i],naive_train_tpr[i]] for i in range(len(naive_train_fpr))]
    naive_test_lst=[[naive_test_fpr[i],naive_test_tpr[i]] for i in range(len(naive_test_fpr))]
    
    
    log_train_VOROS = VOROS.Volume(log_train_lst,[0,1])
    log_test_VOROS = VOROS.Volume(log_test_lst,[0,1])
    
    forest_train_VOROS = VOROS.Volume(forest_train_lst,[0,1])
    forest_test_VOROS = VOROS.Volume(forest_test_lst,[0,1])
    
    naive_train_VOROS = VOROS.Volume(naive_train_lst,[0,1])
    naive_test_VOROS = VOROS.Volume(naive_test_lst,[0,1])
    
    data_dict = {'log_train_auroc': log_train_auroc,
                 'log_train_VOROS': log_train_VOROS,
                 'log_test_auroc' : log_test_auroc,
                 'log_test_VOROS' : log_test_VOROS,
                 'log_train_curve' : log_train_lst,
                 'log_test_curve' : log_test_lst,
                 
                 'forest_train_auroc': forest_train_auroc,
                 'forest_train_VOROS': forest_train_VOROS,
                 'forest_test_auroc' : forest_test_auroc,
                 'forest_test_VOROS' : forest_test_VOROS,
                 'forest_train_curve' : forest_train_lst,
                 'forest_test_curve' : forest_test_lst,
                 
                 'naive_train_auroc': naive_train_auroc,
                 'naive_train_VOROS': naive_train_VOROS,
                 'naive_test_auroc' : naive_test_auroc,
                 'naive_test_VOROS' : naive_test_VOROS,
                 'naive_train_curve' : naive_train_lst,
                 'naive_test_curve' : naive_test_lst,
                }
                 
    return data_dict

if __name__=='__main__':
    state=int(input("Input a random state: "))
    labels_path=input("Input the path to labels.csv: ")
    preds=create_models(labels_path,state)
    log_curve,naive_curve,forest_curve=preds['log_test_curve'],preds['naive_test_curve'],preds['forest_test_curve']
    print(f"Log AUROC: {preds['log_test_auroc']}\nLog VOROS: {preds['log_test_VOROS']}\nNaive AUROC: {preds['naive_test_auroc']}\nNaive VOROS: {preds['naive_test_VOROS']}\nForest AUROC: {preds['forest_test_auroc']}\nForest VOROS: {preds['forest_test_VOROS']}\n")
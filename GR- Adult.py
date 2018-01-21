"""
Created on Mon Dec  4 07:22:49 2017

@author: George L. Roberts

Want to classify whether a person is earning >50k or not
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import time

def main():
    X,realTest = loadData()
    realId=realTest['Id']
    ## Turn the label into a boolean vector
    X.income = X.income.astype(str).str[0] == '>'
    
    y = X.income
    X=X.drop(['income'],axis=1)
    
    realTestLength=len(realTest)
    X=pd.concat([X,realTest],axis=0,ignore_index=True)
    X = transformData(X)
    
    realTest=X[-realTestLength:]
    X=X[:-realTestLength]
    
    #print (train[["fnlwgt", "income"]].groupby(['fnlwgt'], as_index=False).mean())
    #fig, ax = plt.subplots()
    #sns.barplot(ax=ax, x='fnlwgt', y='income', data=train)
    
    XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=0.2)
    
    ### A basic estimator with this data. Running through scikit-learn's
    # best estimator cheatsheet gives a linearSVC as the starting point. 
    estimators = [('clf', GradientBoostingClassifier(max_depth=5))]
    pipe = Pipeline(estimators)
    #parameters ={'clf__C': np.logspace(-4,1,5),
    #             'clf__kernel':['linear','poly','rbf','sigmoid'],
    #}
    #parameters={'clf__n_neighbors':[2,3,4,5]}
    #parameters={'clf__n_estimators':np.linspace(5,30,26,dtype=int)}
    parameters={'clf__learning_rate':np.logspace(-2,0,1)}
    start = time.time()

    gs_clf = GridSearchCV(pipe, parameters)
    
    gs_clf.fit(XTrain,yTrain)
    print(gs_clf.best_params_)
    
    trainScore = gs_clf.best_score_
    testScore = np.mean(cross_val_score(gs_clf,XTest,yTest))
    
    print("Train score is {:.1f}% and test score is {:.1f}%"\
          .format(100*trainScore,100*testScore))
    trainScore = gs_clf.best_score_
    
    testPrediction = pd.DataFrame(gs_clf.predict(realTest), columns=['income'])
    printPrediction=pd.concat([realId,testPrediction],axis=1)
    printPrediction.to_csv("Predictions.csv", index=False)
    
    end = time.time()
    print(end - start) 
    
def transformData(X):
    """ Transform the features into usable features """
    X=X.drop(['Id'],axis=1)
    
     ## fnlwgt - Bin into quartiles: turns out to be not very descriptive
    bins = [0,X.fnlwgt.quantile(q=0.25), X.fnlwgt.quantile(q=0.5), \
            X.fnlwgt.quantile(q=0.75), X.fnlwgt.quantile(q=1)]
    group_names = ['LQ', 'MQ', 'UQ','Max']
    X.fnlwgt = pd.cut(X.fnlwgt, bins, labels=group_names)
    X.fnlwgt= X.fnlwgt.astype(str) # Cast back from category for label encoding

    ## Label encoder
    Xobj = X.select_dtypes(include=['object']).copy()
    Xval = X.select_dtypes(exclude=['object']).copy()
    le=LabelEncoder()
    for column in Xobj:
        if column !='Id':
            le.fit(Xobj[column])
            Xobj[column]=le.transform(Xobj[column])
    X = pd.concat([Xval,Xobj],axis=1)
    
    return X

def loadData():
    """Load the titanic dataset into memory, including the test data"""
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    return train,test


if __name__=='__main__':
    main()
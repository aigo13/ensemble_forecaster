# -*- coding: utf-8 -*-
"""
Support Vector regression을 기반으로한 리그레서
Created on Tue Jun 28 09:05:31 2022

@author: SE19078
"""

from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

class BaseSVR():
    """
        SVR Base Learner
    """
    
    def __init__(self):        
        #self.scaler = StandardScaler()
        self.scaler = RobustScaler()
        # alpha_1/2, lambda_1/2  -> Grid Search 필요
        self.learner = SVR() # parameter 세팅?
        
    def fit(self, x, Y):
        """
        Support Vector Regressor를 fitting

        Parameters
        ----------
        x : Array (n_samples, n_features)
            
        Y : Array (n_samples, n_outputs)            

        Returns
        -------
        Number
            R^2 error

        """
        # scale data
        self.scaler = self.scaler.fit(x)
        xx = self.scaler.transform(x)
        self.fitted = self.learner.fit(xx, Y)
        return self.fitted.score(xx, Y) # R^2 값 return
        
    def predict(self, x):
        """
        fitting 결과를 이용하여 x에 대한 값을 prediction

        Parameters
        ----------
        x : Array (n_samples, n_features)            

        Returns
        -------
        Array (n_samples, n_outputs)

        """
        xx = self.scaler.transform(x)
        return self.fitted.predict(xx)
    

#### Test
if __name__ == "__main__":
    # 데이터 읽어오기
    df = pd.read_csv('../data/20030101_20220627_k200ewm.csv')    
    df = df.dropna()
    df = df.set_index('STD_DT2')
    
    all_cols = np.array(df.columns)
    target_cols = np.array(['F_K200_RET_SUM', 'F_K200_RET_STD'])
    #feature_cols = np.setdiff1d(all_cols, target_cols)
    
    # forward return sum과 std를 만들어줌
    fwdrt = np.array(df['K200_RET_SUM'][20:])
    fwdstd = np.array(df['K200_RET_STD'][20:])
    tmpdf = pd.DataFrame()
    tmpdf['F_K200_RET_SUM'] = fwdrt
    tmpdf['F_K200_RET_STD'] = fwdstd
    tmpdf = tmpdf.set_index(df.index[:-20])
    
    # Y는 (n_samples, 1)만 가능
    x_train = np.array(df[all_cols][:-30])
    y_train = np.array(tmpdf['F_K200_RET_SUM'][:-10])
    
    x_test = np.array(df[all_cols][-30:])
    y_test = tmpdf['F_K200_RET_SUM'][-10:]
    
    learner = BaseSVR()
    r2score = learner.fit(x_train, y_train)
    predicted = learner.predict(x_test)
    
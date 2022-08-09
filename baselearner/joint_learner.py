# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 08:52:21 2022

단일 결과만 내는 알고리즘을 조합하여 원하는 수의 output을 만들어냄
"""
from typing import List, Any, Dict
import numpy as np

class JointLearner():
    
    # learners
    learners: List[Any]
    params: List[Dict] # parameter는 개별 learner의 parameter list
    
    def __init__(self, n_output, learners):
        # 갯수 체크
        if len(learners) != n_output:
            raise Exception(f"output 갯수와 learner의 수가 같아야 합니다. output 수 : {n_output}")
        
        self.n_output = n_output
        
        # err check
        for a_learner in learners:
            if not (hasattr(a_learner, "fit") and hasattr(a_learner, "predict")):
                raise TypeError(f"fit 또는 predict가 없습니다. - {a_learner}")
        self.learners = learners
    
    
    def fit(self, X, y, sample_weight=None):
        """
        등록된 learner의 fit 함수를 호출하고 결과를 저장함.

        Parameters
        ----------
        X : Array-like
            X feature
        y : Array-like
            대상이 되는 feature
        sample_weight : Array-like, optional
            Sample 별 가중치

        Returns
        -------
        JointLearner
            fitting된 learner

        """
        # 타입 변환
        if not isinstance(X, np.ndarray):
            xx = np.array(X)
        else:
            xx = X
        
        if not isinstance(y, np.ndarray):
            yy = np.array(y)
        else:
            yy = y
            
        # learner의 fit
        fitted = []
        for a_learner in self.learners:
            fitted_l = a_learner.fit(X=xx, y=yy, sample_weight=sample_weight)
            fitted.append(fitted_l)
        
        # 결과 저장
        self.learners = fitted
        
        return self
    
    def predict(self, X):
        """
        등록한 learner들의 predict를 호출하고 결과를 종합하여 줌

        Parameters
        ----------
        X : Array-like
            prediction에 사용할 feature

        Returns
        -------
        y_pred : Array-like
            결과 값

        """
        if not isinstance(X, np.ndarray):
            xx = np.array(X)
        else:
            xx = X
            
        out_list = []
        for a_learner in self.learners:
            y = a_learner.predict(xx)
            out_list.append(y)
        
        yy = np.array(out_list).T # transpose
        
        return yy
    
    def score(self, X, y, sample_weight=None):
        """
        R2 score를 구함

        Parameters
        ----------
        X : Array-like
            prediction에 사용할 feature
        y : Array-like
            true 값
        sample_weight : array-like, optional
            sample 별 weight. The default is None.

        Returns
        -------
        r2 : TYPE
            DESCRIPTION.

        """
        # predict 호출
        y_pred = self.predict(X)
        
        # score 계산
        if not isinstance(y, (np.ndarray)):
            y_true = np.array(y)
        else:
            y_true = y
        
        u = ((y_true - y_pred)**2).sum()
        v = ((y_true - y_true.mean())**2).sum()
        r2 = 1. - u/v

        return r2
    
#def set_params(params_list):
#   for a_param, learner in zip(params_list, self.learners):
            

# if __name__ == "__main__":
#     # test
#     from sklearn.linear_model import Ridge
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.pipeline import Pipeline
    
#     a_joint = JointLearner(n_output=2, learners=[Ridge(), Ridge()])
#     a_pipe = Pipeline(steps=[('scaler', StandardScaler()), 
#                              ('learner', a_joint)])
    
#     X = np.random.rand(3,2)
#     y = np.random.rand(3,2)
#     a_pipe.fit(X, y)
#     a_pipe.predict(X)
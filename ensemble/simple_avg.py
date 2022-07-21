# -*- coding: utf-8 -*-
"""
Simple Average Ensemble model

@author: SE19078
"""

import numpy as np

class SimpleAvgEnsemble():
    
    def __init__(self):
        pass
    
    # 평균내주는 utility 함수
    def _calc_avg(self, X, n_out):
        # numpy array가 아닌 경우 변환해줌(indexing 목적)
        if not isinstance(X, (np.ndarray)):
            xx = np.array(X)
        else:
            xx = X
            
        # index 만들기
        idx_list = []
        for i in range(n_out):
            f_idx = np.arange(start=i, \
                               stop=(self.n_feat-(self.n_out-i-1)), \
                               step=self.n_out)
            idx_list.append(f_idx)
        
        # feature selection하고 feature끼리 평균내기
        # (n_samples, n_out)이 column을 따라 붙어있다고 가정
        y_list= []
        for idx_arr in idx_list:
            xx1 = xx[:, idx_arr]
            y1 = np.apply_along_axis(np.mean, 1, xx1)
            y_list.append(y1)
            
        # 결과 만들기    
        y_fitted = np.array(y_list).T        
        return y_fitted
    
    
    def fit(self, X, y, **fit_args):
        self.n_samples = len(X)
        self.n_feat = len(X[0]) # featur 수
        self.n_out = len(y[0]) # target이 되는 y의 column 수
        
        # error 체크
        if len(y) != self.n_samples:
            raise RuntimeError("X와 y의 sample 수가 다릅니다.")
        
        if self.n_feat % self.n_out != 0:
            raise RuntimeError("Ensemble 대상이 되는 feature는 output의 배수여야 합니다.")
                
        # 결과 만들기    
        self.y_fitted = self._calc_avg(X, self.n_out)        
        return self
    
    def predict(self, X, **pred_args):
        if self.n_feat != len(X[0]):
            raise RuntimeError("Feature수가 맞지 않습니다.")
        
        # average        
        y_pred = self._calc_avg(X, self.n_out)
        return y_pred
    
    def score(self, X, y, sample_weight=None):
        if self.n_feat != len(X[0]):
            raise RuntimeError("Feature수가 맞지 않습니다.")
        
        if not isinstance(y, (np.ndarray)):
            yy = np.array(y)
        else:
            yy = y
            
        y_hat = self._calc_avg(X, self.n_out)
        # r2 score 계산
        u = ((yy - y_hat)**2).sum()
        v = ((yy - yy.mean())**2).sum()
        r2 = 1. - u/v
        return r2

################### TEST #########################
# if __name__ == "__main__":
#     test_x = []
#     for i in range(10):
#         test_x.append([1,2,3,4,5,6])
    
#     test_y = []
#     for i in range(10):
#         test_y.append([2.5, 3.5, 4.5])
    
#     se = SimpleAvgEnsemble()
#     se = se.fit(test_x, test_y)
#     print('-- fitted --')
#     print(se.y_fitted)
    
#     y_pred = se.predict(test_x)
#     print('-- predicted --')
#     print(se.y_fitted)
    
#     r2 = se.score(test_x, test_y)
#     print('-- score --')
#     print(r2)
    
        
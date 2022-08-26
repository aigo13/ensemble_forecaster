# -*- coding: utf-8 -*-
"""
    결과를 column끼리 묶어주는 pipeline
    [TODO] GridSearch를 위해 fit_parameter 적용하는 것 만들어야 함.
           endemble method만 해당. pipeline쪽은 model 클래스에서
"""
from typing import List, Any
import numpy as np

class UnionPipe:
    """
        pipeline상의 learner들의 결과를 column by column으로 묶어줌
        묶어진 결과를 마지막 learner의 input으로 넣어줌
    """
    # base learner pipeline이 등록되어있는 list
    # parallel 수행
    learners: List[Any]
    
    def __init__(self, learners):
        """
        init

        Parameters
        ----------
        data : pandas DataFrame
            
        learners : list of tuples
            (name, features to use, pipeline)

        Returns
        -------
        None.

        """
        self.learners = learners        
    
    
    def _iter(self):
        """
        base learner pipeline을 index, name, target X, learner 로 보내줌

        Yields
        ------
        idx : integer
            learner의 index
        name : string
            learner의 이름
        learner : Pipeline또는 object
            base learner
        """
        for idx, (name, x_feat, learner) in enumerate(self.learners):
            yield idx, name, x_feat, learner
    
    def fit(self, data, y, **fit_params):
        """
        저장된 learner(pipeline)의 fit을 호출
        동작은 해당 learner의 동작을 따름        

        Returns
        -------
        UnionPipe object
        """
        # [TODO] parallel?
        for idx, name, x_feat, learner in self._iter():            
            X = np.array(data[x_feat])
            #print(x_feat)
            #print(data[x_feat].columns)
            assert len(x_feat) == len(X[0]) # for error check
            fitted_learner = learner.fit(X, y)
            self.learners[idx] = (name, x_feat, fitted_learner) # learner update
            
        return self
    
    def score(self, X, y):
        """
        fit score를 리턴함

        Returns
        -------
        scores : List
            (name, score)의 List

        """
        scores = []
        for idx, name, x_feat, learner in self._iter():
            x_pipe = np.array(X[x_feat])
            assert len(x_feat) == len(x_pipe[0]) # for error check
            a_score = learner.score(x_pipe, y)
            scores.append((name, a_score))
        
        return scores
            
    
    def predict(self, data):
        """
        저장된 learner(pipeline)의 predict를 호출
        동작은 해당 learner의 동작을 따름

        Parameters
        ----------
        X : pandas DataFrame with (n_samples, n_features)
            X data
        Returns
        -------
        predicted y
        """        
        y_pred = []
        y_len = len(data)
        for idx, name, x_feat, learner in self._iter():
            xx = np.array(data[x_feat])
            assert len(x_feat) == len(xx[0]) # for error check       
            y_l = learner.predict(xx) # ndarray return 가정
            y_l = y_l.reshape(y_len, -1) # reshape (n_samples, n_output)
            y_pred.append(y_l) # learner별 결과를 행으로 붙여줌        
        
        # 결과를 컬럼으로 붙여주기
        # [TODO] 좀 더 효율적으로 할 수 있는 방법은?
        y_ret = y_pred[0]
        for i in range(1, len(y_pred)):
            y_ret = np.append(y_ret, y_pred[i], axis=1)        
        return y_ret
    
    
################# Test code ##################
# class DummyPipe:
#     mult : float
    
#     def __init__(self, m):
#         self.mult = m
    
#     def fit(self, X, y=None):
#         self.y_pred = np.array(X) * self.mult
#         return self
    
#     def predict(self, X):
#         return self.y_pred

# if __name__ == "__main__":
#     import pandas as pd
#     X = np.array([1, 2, 3, 4, 5, 6])
#     X = np.reshape(X, (3, 2))
#     Xdf = pd.DataFrame(X, columns=['a', 'b'])    
#     YY = np.array([1,2,3])
#     targets = [['a', 'b'], ['a', 'b'], ['a', 'b']]
#     learner_list = []
#     learner_list.append(DummyPipe(1.0))
#     learner_list.append(DummyPipe(2.0))
#     learner_list.append(DummyPipe(3.0))
    
#     nm = ['learner_' + str(x+1) for x in range(3)]
#     steps = [x for x in zip(nm, targets, learner_list)]
#     print(steps)
#     print(X)
#     up = UnionPipe(steps)
    
#     up.fit(data=Xdf, y=YY)
#     y_pred = up.predict(Xdf)
#     print(y_pred)
    
    
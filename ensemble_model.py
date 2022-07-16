"""
    전체 예측모형을 wrapping한 클래스
"""

import numpy as np
from sklearn.pipeline import Pipeline

from ensemble import UnionPipe

class EnsembleModel:
    """
        Data 및 model을 연결하는 pipeline
    """

    def __init__(self, data, target):        
        """Data  및 Model 연결을 위한  pipeline 생성 클래스

        Args:
            data (pandas.DataFrame): 학습에 사용할  data
            target (array-like): 학습 타겟의 되는 컬럼 명

        Raises:
            Exception: target이 list, tuple, ndarray가 아닐경우
        """
        if isinstance(target, (list, tuple, np.ndarray)) == False:
            raise TypeError("target은 Array 형태여야 합니다.(list, tuple)")
        self.data = data.copy() # pandas dataframe copy
        self.target = target  
        self.y_true = np.array(data[target]) # 실제 y data
        self.pipelines = {} # p_name: pipeline object  
        self.union_pipe = None
        self.ensemble = None
        # (n_samples, n_out)형태의 뭉쳐져있음.
        #self.em_params = { 'col_aggr' : True } 

    def add_base_pipe(self, p_name, scalers, learners, features=None):
        """
        주어진 base learners를 사용하여 p_name을 갖는 pipeline 정보 생성

        Args:
            p_name (string): 생성할 pipleline의 이름
            scalers (list): 데이터 바로 뒤에 사용할 scaler. 리스트 순서대로 데이터 통과.
            learners (list): Scaler 바로 뒤에 위치할 base learner들. 리스트 순서대로 데이터 통과
            features (array-like, optional): 학습에 사용할 column명. 주어지지 않을 경우 모두 사용. 
                                             Defaults to None.            
        Returns:
            self
        """               

        # 최종 (features, pipline) 형태의 tuple을 만들어 self.pipelines에 p_name : tuple을 넣음
        # features와 target
        if features is not None:
            if not isinstance(features, (list, tuple, np.ndarray)):
                raise TypeError("features는 string의 array-like 형태여야 합니다.")
            
            x_feat = np.array(features)
        else:
            
            x_feat = np.setdiff1d(np.array(self.data.columns), np.array(features))
        
        # scaler와 base learner -> 현재 scaler->base learner의 순서만 지원
        n_scaler = len(scalers)
        n_learners = len(learners)
        scaler_nm = ['scaler_' + str(x+1) for x in range(n_scaler)]
        lerners_nm = ['base_' + str(x+1) for x in range(n_learners)]

        steps = []
        # scaler 추가
        for pair in zip(scaler_nm, scalers):
            steps.append(pair)
        
        # base lerner 추가
        for pair in zip(lerners_nm, learners):
            steps.append(pair)
        
        a_pipe = Pipeline(steps=steps)
        # 추가
        self.pipelines[p_name] = (x_feat, a_pipe)
        return self


    def build_ensemble(self, ensemble):
        """
        최종 ensemble 모형을 만듦

        Args:
            ensemble (object): Ensemble모형의 오브젝트
        
        Returns:
            self
        """       
        
        # union pipeline 만들기
        learners = []
        for a_key in self.pipelines.keys():
            x_feat = self.pipelines[a_key][0]            
            a_learner = self.pipelines[a_key][1] # learner object
            learners.append((a_key, x_feat, a_learner))
        
        self.union_pipe = UnionPipe(data=self.data, learners=learners)
        
        # attribute check
        if (hasattr(ensemble, 'fit') and 
            hasattr(ensemble, 'predict')) == False:
            raise TypeError("ensemble객체에 fit 함수와 predict 함수가 있어야 합니다.")
            
        self.ensemble = ensemble
        
        return self
    
    def fit(self):
        """
        주어진 data에 맞도록 모형 fit
        하위 pipeline의 fit을 호출한다.

        Returns
        -------
        EnsembleModel
            fitted object

        """
        # ensemble pipeline의 fit을 호출
        self.union_pipe = self.union_pipe.fit(data=self.data, y=self.y_true)
        # error 계산을 위한 x data predict
        self.y_fitted = self.union_pipe.predict(data=self.data)    
        # Ensemble model의 fit 호출
        self.ensemble = self.ensemble.fit(X=self.y_fitted, y=self.y_true)
        
        return self
        
    def predict(self, data):
        """
        Prediction 수행

        Parameters
        ----------
        data : pandas DataFrame
            예측을 위한 feature 데이터를 가지고 있는 데이터프레임

        Returns
        -------
        y_pred : Array-like
            (n_samples, n_outputs) 형태의 prediction 결과

        """
        # ensemble pipeline의 predict를 호출
        y_pred = self.union_pipe.predict(data=data)        
        # Ensemble model의 predict 호출
        y_pred = self.ensemble.predict(X=y_pred)
        # predict 결과 return
        return y_pred
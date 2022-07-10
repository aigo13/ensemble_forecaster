"""
    Pipeline class
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


class EnsembleModel():
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
            raise Exception("target은 Array 형태여야 합니다.(list, tuple, np.ndarray)")
        self.data = data.copy() # pandas dataframe copy
        self.target = target
        self.base_learner = []
        self.pipelines = {} # p_name: pipeline object        
        self.ensemble = None
        pass 

    def make_base_pipe(self, p_name, scalers, learners, features=None, target=None):
        """
        주어진 base learners를 사용하여 p_name을 갖는 pipeline 생성

        Args:
            p_name (string): 생성할 pipleline의 이름
            scalers (list): 데이터 바로 뒤에 사용할 scaler. 리스트 순서대로 데이터 통과.
            learners (list): Scaler 바로 뒤에 위치할 base learner들. 리스트 순서대로 데이터 통과
            features (array-like, optional): 학습에 사용할 column명. 주어지지 않을 경우 모두 사용. 
                                             Defaults to None.
            target (array-like, optional): prediction target에 사용할 column명. 주어지지 않을 경우 
                                          클래스 초기화 시 주어진 값 사용. Defaults to None.            
        """               

        # 최종 (features, target, pipline) 형태의 tuple을 만들어 self.pipelines에 p_name : tuple을 넣음
        pass

    def build_ensemble(self, ensemble):
        """
        최종 ensemble 모형을 만듦

        Args:
            ensemble (object): Ensemble모형의 오브젝트
        """

        # FeatureUnion을 사용해서 앞서 만든 BasePipe를 모두 연결함.
        pass
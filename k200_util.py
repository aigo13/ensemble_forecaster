"""
    Utility functions
"""

import pandas as pd
from datagen import embed_ts
# scaler
from sklearn.preprocessing import StandardScaler
# base models
from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge # (n_samples, n_target)
from sklearn.ensemble import RandomForestRegressor

from sklearn.multioutput import MultiOutputRegressor
# Multioutput으로 Wrapping 필요
from xgboost import XGBRegressor
from sklearn.svm import SVR # (n_samples, )
from sklearn.linear_model import TweedieRegressor
# from sklearn.tree import DecisionTreeRegressor # (n_samples, n_target)
# from sklearn.linear_model import GammaRegressor # (n_samples) -> GLM
# from sklearn.gaussian_process import GaussianProcessRegressor # (target의 mean/std)

# KOSPI200 feature 모음
_k200_feat_dict = {
    'KOSPI2': ['KOSPI2'],
    'RET' : ['KOSPI2_RET', 'KOSPI2_RET_AVG', 'KOSPI2_RET_MIN', 'KOSPI2_RET_MAX', 
             'KOSPI2_RET_AMP', 'KOSPI2_RET_STD', 'KOSPI2_RET_SKEW', 'KOSPI2_RET_SUM'],
    'VOL' : ['VKOSPI', 'VSPREAD'],
    'FX' : ['USDKRW', 'USDKRW_V', 'FX_RET', 'FX_RET_AVG', 'FX_RET_MIN',
            'FX_RET_MAX', 'FX_RET_AMP', 'FX_RET_STD', 'FX_RET_SKEW', 'FX_RET_SUM'],
    'COM' : ['CRUDE_F', 'CRUDE_RET', 'CRUDE_RET_AVG', 'CRUDE_RET_MIN',
             'CRUDE_RET_MAX', 'CRUDE_RET_AMP', 'CRUDE_RET_STD', 'CRUDE_RET_SKEW',
             'CRUDE_RET_SUM'],
    'FIDX': ['SPX', 'VSPX', 'SPX_RET', 'SPX_RET_AVG', 'SPX_RET_MIN', 'SPX_RET_MAX',
             'SPX_RET_AMP', 'SPX_RET_STD', 'SPX_RET_SKEW', 'SPX_RET_SUM',
             'VIX_SPREAD'],
    'CDS': ['ROKCDS']     
     }

_embbed_target = ('KOSPI2', 'KOSPI2_RET', 'VKOSPI', 'USDKRW', 'USDKRW_V', 'FX_RET', 
                'CRUDE_F', 'CRUDE_RET', 'SPX', 'VSPX', 'SPX_RET', 'ROKCDS')

# time seriese embedding
# embedding으로 추가된 column의 dict를 돌려준다.
def embed_data(df : pd.DataFrame, target_col : tuple, eb_size : int):
    # time series data를 embedding하기
    col_list = {}
    for a_col in target_col:
        # embedding
        edf = embed_ts(df, a_col, eb_size)
        # 가져온 데이터를 기존 dataframe에 넣어줌        
        edf = edf.set_index(df.index[(eb_size-1):])        
        col_list[a_col] = edf.columns.to_list()
        df = pd.concat([df, edf], axis=1)    
    
    return (df, col_list)

# Ridge Regressor 계열 base learner 추가
def add_ridge_based_pipe(ensemble, ts_embed):
    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯
    prefix = "RIDGE"    
    alpha = 0.1
    
    # 1번 Vol
    p_name = "_".join([prefix, "01"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        
    #print(f'--> adding {p_name} with features ---> RET, VOL')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge(alpha=alpha)], features=feat)
    
    # 2번 FX
    p_name = "_".join([prefix, "02"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        
    #print(f'--> adding {p_name} with features ---> FX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge(alpha=alpha)], features=feat)    
    
    # 3번 WTI
    p_name = "_".join([prefix, "03"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['COM'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        
    #print(f'--> adding {p_name} with features ---> WTI')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge(alpha=alpha)], features=feat)    
    
    # 4번 S&P500
    p_name = "_".join([prefix, "04"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FIDX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        
    #print(f'--> adding {p_name} with features ---> SPX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge(alpha=alpha)], features=feat)
    
    # 5번 CDS
    p_name = "_".join([prefix, "05"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> CDS')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge(alpha=alpha)], features=feat)    
    
    # 6번 FULL
    p_name = "_".join([prefix, "06"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    feat.extend(_k200_feat_dict['FX'])
    feat.extend(_k200_feat_dict['FIDX'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])          
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> Full')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge(alpha=alpha)], features=feat)    
    print(f'--> added {prefix} based learners with features')  

# Kernel Ridge Regressor 계열 base learner 추가
def add_kridge_based_pipe(ensemble, ts_embed):
    # RBF kernel을 사용하는 kernel ridge
    prefix = "KRIDGE"    
    a = 0.05

    # 1번 Vol
    p_name = "_".join([prefix, "01"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        
    #print(f'--> adding {p_name} with features ---> RET, VOL')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [KernelRidge(alpha=a, kernel=RBF())], features=feat)
    
    # 2번 FX
    p_name = "_".join([prefix, "02"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        
    #print(f'--> adding {p_name} with features ---> FX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [KernelRidge(alpha=a, kernel=RBF())], features=feat)    
    
    # 3번 WTI
    p_name = "_".join([prefix, "03"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['COM'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        
    #print(f'--> adding {p_name} with features ---> WTI')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [KernelRidge(alpha=a, kernel=RBF())], features=feat)    
    
    # 4번 S&P500
    p_name = "_".join([prefix, "04"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FIDX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        
    #print(f'--> adding {p_name} with features ---> SPX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [KernelRidge(alpha=a, kernel=RBF())], features=feat)
    
    # 5번 CDS
    p_name = "_".join([prefix, "05"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> CDS')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [KernelRidge(alpha=a, kernel=RBF())], features=feat)    
    
    # 6번 FULL
    p_name = "_".join([prefix, "06"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    feat.extend(_k200_feat_dict['FX'])
    feat.extend(_k200_feat_dict['FIDX'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])          
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> Full')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [KernelRidge(alpha=a, kernel=RBF())], features=feat) 
    print(f'--> added {prefix} based learners with features')  

# Random Forest Regressor 계열 base learner 추가
def add_rf_based_pipe(ensemble, ts_embed):
    # RBF kernel을 사용하는 kernel ridge
    prefix = "RFR" 
    min_samples = 5
    #min_samples = 1
    
    # 1번 Vol
    p_name = "_".join([prefix, "01"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        
    #print(f'--> adding {p_name} with features ---> RET, VOL')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [RandomForestRegressor(min_samples_leaf=min_samples)], 
                                features=feat)
    
    # 2번 FX
    p_name = "_".join([prefix, "02"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        
    #print(f'--> adding {p_name} with features ---> FX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [RandomForestRegressor(min_samples_leaf=min_samples)], 
                                features=feat)    
    
    # 3번 WTI
    p_name = "_".join([prefix, "03"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['COM'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        
    #print(f'--> adding {p_name} with features ---> WTI')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [RandomForestRegressor(min_samples_leaf=min_samples)], 
                                features=feat)    
    
    # 4번 S&P500
    p_name = "_".join([prefix, "04"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FIDX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        
    #print(f'--> adding {p_name} with features ---> SPX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [RandomForestRegressor(min_samples_leaf=min_samples)], 
                                features=feat)
    
    # 5번 CDS
    p_name = "_".join([prefix, "05"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> CDS')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [RandomForestRegressor(min_samples_leaf=min_samples)], 
                                features=feat)    
    
    # 6번 FULL
    p_name = "_".join([prefix, "06"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    feat.extend(_k200_feat_dict['FX'])
    feat.extend(_k200_feat_dict['FIDX'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])          
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> Full')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [RandomForestRegressor(min_samples_leaf=min_samples)], 
                                features=feat) 
    print(f'--> added {prefix} based learners with features')  

# XGBoost 계열 base learner 추가
# MultiOutput으로 wrapping
def add_xgb_based_pipe(ensemble, ts_embed):
    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯
    prefix = "XGB"    
    alpha = 0.5
    max_depth = 5
    
    # 1번 Vol
    p_name = "_".join([prefix, "01"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        
    #print(f'--> adding {p_name} with features ---> RET, VOL')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(XGBRegressor(alpha=alpha, max_depth=max_depth))], 
                                features=feat)
    
    # 2번 FX
    p_name = "_".join([prefix, "02"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        
    #print(f'--> adding {p_name} with features ---> FX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(XGBRegressor(alpha=alpha, max_depth=max_depth))], 
                                features=feat)
    
    # 3번 WTI
    p_name = "_".join([prefix, "03"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['COM'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        
    #print(f'--> adding {p_name} with features ---> WTI')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(XGBRegressor(alpha=alpha, max_depth=max_depth))], 
                                features=feat)
    
    # 4번 S&P500
    p_name = "_".join([prefix, "04"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FIDX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        
    #print(f'--> adding {p_name} with features ---> SPX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(XGBRegressor(alpha=alpha, max_depth=max_depth))], 
                                features=feat)
    
    # 5번 CDS
    p_name = "_".join([prefix, "05"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> CDS')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(XGBRegressor(alpha=alpha, max_depth=max_depth))], 
                                features=feat)
    
    # 6번 FULL
    p_name = "_".join([prefix, "06"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    feat.extend(_k200_feat_dict['FX'])
    feat.extend(_k200_feat_dict['FIDX'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])          
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> Full')  
    ensemble.add_base_pipe(p_name, [StandardScaler()],
                                [MultiOutputRegressor(XGBRegressor(alpha=alpha, max_depth=max_depth))], 
                                features=feat)
    print(f'--> added {prefix} based learners with features')

# SVR based pipe
def add_svr_based_pipe(ensemble, ts_embed):
    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯
    prefix = "SVR"
    c_v = 0.1    
    
    # 1번 Vol
    p_name = "_".join([prefix, "01"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        
    #print(f'--> adding {p_name} with features ---> RET, VOL')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(SVR(C=c_v))], 
                                features=feat)
    
    # 2번 FX
    p_name = "_".join([prefix, "02"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        
    #print(f'--> adding {p_name} with features ---> FX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(SVR(C=c_v))], 
                                features=feat)
    
    # 3번 WTI
    p_name = "_".join([prefix, "03"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['COM'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        
    #print(f'--> adding {p_name} with features ---> WTI')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(SVR(C=c_v))], 
                                features=feat)
    
    # 4번 S&P500
    p_name = "_".join([prefix, "04"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FIDX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        
    #print(f'--> adding {p_name} with features ---> SPX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(SVR(C=c_v))], 
                                features=feat)
    
    # 5번 CDS
    p_name = "_".join([prefix, "05"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> CDS')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(SVR(C=c_v))], 
                                features=feat)
    
    # 6번 FULL
    p_name = "_".join([prefix, "06"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    feat.extend(_k200_feat_dict['FX'])
    feat.extend(_k200_feat_dict['FIDX'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])          
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> Full')  
    ensemble.add_base_pipe(p_name, [StandardScaler()],
                                [MultiOutputRegressor(SVR(C=c_v))],
                                features=feat)
    print(f'--> added {prefix} based learners with features')  

# GLM 계열 base learner 추가(gamma distribution)
# MultiOutput으로 wrapping -> negative target 처리 힘들 듯
def add_glm_based_pipe(ensemble, ts_embed):
    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯
    prefix = "GLM"    
    p = 2 # power 2: gamma distribution
    l = "log" # log for non-negative
    
    # 1번 Vol
    p_name = "_".join([prefix, "01"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        
    #print(f'--> adding {p_name} with features ---> RET, VOL')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(TweedieRegressor(power=p, link=l))], 
                                features=feat)
    
    # 2번 FX
    p_name = "_".join([prefix, "02"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        
    #print(f'--> adding {p_name} with features ---> FX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(TweedieRegressor(power=p, link=l))], 
                                features=feat)
    
    # 3번 WTI
    p_name = "_".join([prefix, "03"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['COM'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        
    #print(f'--> adding {p_name} with features ---> WTI')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(TweedieRegressor(power=p, link=l))], 
                                features=feat)
    
    # 4번 S&P500
    p_name = "_".join([prefix, "04"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['FIDX'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        
    #print(f'--> adding {p_name} with features ---> SPX')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(TweedieRegressor(power=p, link=l))], 
                                features=feat)
    
    # 5번 CDS
    p_name = "_".join([prefix, "05"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> CDS')  
    ensemble.add_base_pipe(p_name, [StandardScaler()], 
                                [MultiOutputRegressor(TweedieRegressor(power=p, link=l))], 
                                features=feat)
    
    # 6번 FULL
    p_name = "_".join([prefix, "06"])
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])
    feat.extend(_k200_feat_dict['FX'])
    feat.extend(_k200_feat_dict['FIDX'])
    feat.extend(_k200_feat_dict['CDS'])
    # embedding 값들이 있을 경우 해당 값도 포함
    if ts_embed == True:
        feat.extend(_k200_feat_dict['KOSPI2_em'])
        feat.extend(_k200_feat_dict['KOSPI2_RET_em'])          
        feat.extend(_k200_feat_dict['VKOSPI_em'])
        feat.extend(_k200_feat_dict['USDKRW_em'])
        feat.extend(_k200_feat_dict['FX_RET_em'])
        feat.extend(_k200_feat_dict['USDKRW_V_em'])
        feat.extend(_k200_feat_dict['CRUDE_F_em'])
        feat.extend(_k200_feat_dict['CRUDE_RET_em'])
        feat.extend(_k200_feat_dict['SPX_em'])
        feat.extend(_k200_feat_dict['SPX_RET_em'])
        feat.extend(_k200_feat_dict['VSPX_em'])
        feat.extend(_k200_feat_dict['ROKCDS_em'])        
        
    #print(f'--> adding {p_name} with features ---> Full')  
    ensemble.add_base_pipe(p_name, [StandardScaler()],
                                [MultiOutputRegressor(TweedieRegressor(power=p, link=l))], 
                                features=feat)
    print(f'--> added {prefix} based learners with features')
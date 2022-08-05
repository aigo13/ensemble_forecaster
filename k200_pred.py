# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:49:51 2022

@author: SE19078
"""

import pandas as pd
import numpy as np

# scaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# ensemble models
from ensemble import MyEnsembleModel
from ensemble import SimpleAvgEnsemble

# base models
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge # (n_samples, n_target)
from sklearn.svm import SVR # (n_samples, )
from sklearn.tree import DecisionTreeRegressor # (n_samples, n_target)
from sklearn.linear_model import GammaRegressor # (n_samples) -> GLM
from sklearn.gaussian_process import GaussianProcessRegressor # (target의 mean/std)
from sklearn.ensemble import RandomForestRegressor
#import xgboost

# data and utility
from datagen import load_k200_data
from datagen import embed_ts

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


# kospi200 data 불러오기
def load_data_k200(start_dt, end_dt, rolling_win=20, from_file=False, path=None):    
    if from_file == True and path is not None:
        df = pd.read_csv(path)
    else:
        df = load_k200_data(start_dt, end_dt, rolling_win=rolling_win, save_file=True)
    
    return df

# target window만큼의 향후 return sum, real vol 계산
# KOSPI2_RET을 사용
def calc_mean_var(df, window, concat=True):
    target = 'KOSPI2_RET'
    mvdf = pd.DataFrame()
    mvdf['T_' + target + "_AVG"] = df[target].rolling(window).mean()
    mvdf['T_' + target + "_STD"] = df[target].rolling(window).std()
    mvdf.dropna(inplace=True)
    
    # 예측치이므로 윈도우만큼 땡겨서 붙여줌
    df_index = df.index[:-(window-1)]
    mvdf = mvdf.set_index(df_index)
    
    if concat == True:
        df = pd.concat([df, mvdf], axis=1)    
        return df
    else:
        return mvdf

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

# data 읽어서 가져오기
def prepare_k200(start_dt, end_dt, rolling_win=20, embed=False):    
    # file version
    df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win, \
                        from_file=True, \
                        path="./data/20030101_20220627_k200ewm.csv")
    # db version
    #df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win)

    if embed == True:
        eb_ret = embed_data(df, _embbed_target, rolling_win)
        df = eb_ret[0]
        eb_dict = eb_ret[1]

    # 최종 cleanse    
    df.dropna(inplace=True)

    if embed == False:
        return (df, None)
    else:
        return (df, eb_dict)
    

# Ridge Regressor 계열 base learner 추가
def add_ridge_based_pipe(ensemble, data_df, ts_embed):
    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯
    prefix = "RIDGE"    
    
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
        
    print(f'--> adding {p_name} with features ---> RET, VOL')  
    main_ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge()], features=feat)
    
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
        
    print(f'--> adding {p_name} with features ---> FX')  
    main_ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge()], features=feat)    
    
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
        
    print(f'--> adding {p_name} with features ---> WTI')  
    main_ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge()], features=feat)    
    
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
        
    print(f'--> adding {p_name} with features ---> SPX')  
    main_ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge()], features=feat)
    
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
        
    print(f'--> adding {p_name} with features ---> CDS')  
    main_ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge()], features=feat)    
    

#### main
if __name__ == "__main__":

    target_win = 5
    ts_embed = True
    # data 불러오기
    print('---> Load KOSPI200 Data')
    prep_ret = prepare_k200('20030101', '20220627', rolling_win=20, embed=ts_embed)
    data_df = prep_ret[0]
    # embedding된 time series가 있는 경우 feat_dict에 포함시켜줌
    if ts_embed == True:
        for a_key in prep_ret[1]:
            n_key = "_".join([a_key, "em"]) # embedded ts postfix
            _k200_feat_dict[n_key] = prep_ret[1][a_key]
    
    
    # target 생성하기
    data_df = calc_mean_var(data_df, window=target_win)    
    target_col = ['T_KOSPI2_RET_AVG', 'T_KOSPI2_RET_STD']

    # main ensemble model
    print('---> Build Ensemble Model')
    main_ensemble = MyEnsembleModel(data_df[:-(target_win-1)], target_col)   

    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯    
    add_ridge_based_pipe(main_ensemble, data_df, ts_embed)   
    
    print('--> Build Ensemble(Simple Avg)')
    main_ensemble.build_ensemble(SimpleAvgEnsemble())
    
    print('---> Call fitting')
    fitted = main_ensemble.fit()
    fitted_y = main_ensemble.y_fitted_all
    
    print('---> predict')
    predicted = main_ensemble.predict(data_df[-(target_win-1):])
    pred_all = main_ensemble.predict(data_df[:-(target_win-1)])

    # [TODO] Score 함수 구현
    # 성능 테스트 및 플로팅 고민해보기

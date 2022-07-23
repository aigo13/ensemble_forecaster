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

# ensemble models
from ensemble import MyEnsembleModel
from ensemble import SimpleAvgEnsemble

# base models
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
#import xgboost

# data and utility
from datagen import load_k200_data
from datagen import embed_ts


# KOSPI200 feature 모음
_k200_feat_dict = {
    'KOSPI2': ['KOSPI2'],
    'RET' : ['KOSPI2_RET', 'KOSPI2_RET_AVG', 'KOSPI2_RET_MIN', 'KOSPI2_RET_MAX', 
             'KOSPI2_RET_AMP', 'KOSPI2_RET_STD', 'KOSPI2_RET_SKEW', 'KOSPI2_RET_SUM'],
    'VOL' : ['VKOSPI', 'KOSPI2_RET_STD', 'VSPREAD'],
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

# data 읽어서 가져오기
def prepare_k200(start_dt, end_dt, target_win=5, rolling_win=20):    
    # file version
    df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win, \
                        from_file=True, \
                        path="./data/20030101_20220627_k200ewm.csv")
    # db version
    #df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win)
    df.dropna(inplace=True)
    return df

#### main
if __name__ == "__main__":

    target_win = 5
    # data 불러오기
    print('---> Load KOSPI200 Data')
    data_df = prepare_k200('20030101', '20220627', target_win=target_win, rolling_win=20)
    data_df = calc_mean_var(data_df, window=target_win)
    target_col = ['T_KOSPI2_RET_AVG', 'T_KOSPI2_RET_STD']

    # main ensemble model
    print('---> Build Ensemble Model')
    main_ensemble = MyEnsembleModel(data_df, target_col)   

    # Ridge 계열 추가
    p_name = 'RIDGE_01'
    feat = _k200_feat_dict['KOSPI2'].copy()
    feat.extend(_k200_feat_dict['RET'])
    feat.extend(_k200_feat_dict['VOL'])  
    print(f'--> adding {p_name} with features ---> {feat}')  
    main_ensemble.add_base_pipe(p_name, [StandardScaler()], [Ridge()], features=feat)    
    
    print('ss')

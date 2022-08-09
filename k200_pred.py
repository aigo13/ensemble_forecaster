# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:49:51 2022

@author: SE19078
"""

import pandas as pd
import numpy as np

# ensemble models
from ensemble import EnsemblePredictor
from ensemble import SimpleAvgEnsemble

# data and utility
from datagen import load_k200_data
import k200_util as ku

# kospi200 data 불러오기
def load_data_k200(start_dt, end_dt, rolling_win=20, from_file=False, path=None):    
    if from_file == True and path is not None:
        df = pd.read_csv(path)
    else:
        df = load_k200_data(start_dt, end_dt, rolling_win=rolling_win, save_file=True)
    
    return df

# target window만큼의 향후 return sum, real vol 계산
# KOSPI2_RET을 사용
def calc_target_val(df, window, concat=True):
    target = 'KOSPI2_RET'
    mvdf = pd.DataFrame()
    #mvdf['T_' + target + "_AVG"] = df[target].rolling(window).mean()
    mvdf['T_' + target + "_SUM"] = df[target].rolling(window).sum()
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
def prepare_k200(start_dt, end_dt, rolling_win=20, embed=False):    
    # file version
    df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win, \
                        from_file=True, \
                        path="./data/20030101_20220627_k200ewm.csv")
    # db version
    #df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win)

    if embed == True:
        eb_ret = ku.embed_data(df, ku._embbed_target, rolling_win)
        df = eb_ret[0]
        eb_dict = eb_ret[1]

    # 최종 cleanse    
    df.dropna(inplace=True)

    if embed == False:
        return (df, None)
    else:
        return (df, eb_dict)
    
#### main
if __name__ == "__main__":

    target_win = 5
    ts_embed = True
    # data 불러오기
    print('---> Load KOSPI200 Data')
    prep_ret = prepare_k200('20030101', '20220627', rolling_win=20, embed=ts_embed)
    data_df = prep_ret[0]
    
    # test용 data_df 자르기
    #data_df = data_df[data_df['STD_DT2'] <= '2020-03-01']
    
    # embedding된 time series가 있는 경우 feat_dict에 포함시켜줌
    if ts_embed == True:
        for a_key in prep_ret[1]:
            n_key = "_".join([a_key, "em"]) # embedded ts postfix
            ku._k200_feat_dict[n_key] = prep_ret[1][a_key]
    
    
    # target 생성하기(target_win 동안의 sum과 std)
    data_df = calc_target_val(data_df, window=target_win)    
    target_col = ['T_KOSPI2_RET_SUM', 'T_KOSPI2_RET_STD']

    # main ensemble model
    print('---> Build Ensemble Model')
    main_ensemble = EnsemblePredictor(data_df[:-(target_win-1)], target_col)   

    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯    
    ku.add_ridge_based_pipe(main_ensemble, data_df, ts_embed)
    # kernel-ridge with RBF
    ku.add_kridge_based_pipe(main_ensemble, data_df, ts_embed)    
    # RandomForest 계열 추가
    ku.add_rf_based_pipe(main_ensemble, data_df, ts_embed)
    # SVR 계열 추가
    ku.add_svr_based_pipe(main_ensemble, data_df, ts_embed)
    # XGBoost 계열 추가
    ku.add_xgb_based_pipe(main_ensemble, data_df, ts_embed)    
    
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
    r2 = main_ensemble.score(data_df[:-(target_win-1)], data_df[target_col][:-(target_win-1)])
    pipe_scores = main_ensemble.pipe_score(data_df[:-(target_win-1)], data_df[target_col][:-(target_win-1)])
    print(f'total r2 score : {r2}')
    print(f'individual scores : {pipe_scores}')
    

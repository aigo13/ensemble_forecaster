# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:49:51 2022

@author: SE19078
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt

# ensemble models
from ensemble import EnsemblePredictor
from ensemble import SimpleAvgEnsemble

# data and utility
from datagen import load_k200_data
import k200_util as ku

# kospi200 data 불러오기
def load_data_k200(start_dt, end_dt, rolling_win=20, from_file=False, path=None):    
    if from_file == True and path is not None:
        df = pd.read_csv(path, index_col=0)
        # index를 datetime으로 변환
        new_idx = np.array([dt.strptime(dstr, '%Y-%m-%d') for dstr in df.index])
        df = df.set_index(new_idx)
        # data slice 하기
        start_d = dt.strptime(start_dt, '%Y%m%d')
        end_d = dt.strptime(end_dt, '%Y%m%d')
        df = df[(df.index >= start_d) & (df.index <= end_d)]
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

def load_data(data_start, data_end, rolling_win, target_win, embed_ts):
    # data 불러오기
    print('---> Load KOSPI200 Data')
    prep_ret = prepare_k200(data_start, data_end, 
                            rolling_win=rolling_win, embed=embed_ts)
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
    
    return (data_df, target_col)


# ensemble model을 만들고 fitting하기
def fit_predictor(data_df, fit_start, fit_end, target_win, target_col, ts_embed):    

    start_d = dt.strptime(fit_start, '%Y%m%d')
    end_d = dt.strptime(fit_end, '%Y%m%d')
    
    # data slice
    target_df = data_df[(data_df.index >= start_d) & (data_df.index <= end_d)]
    
    # main ensemble model
    print('---> Build Ensemble Model')
    #main_ensemble = EnsemblePredictor(target_df[:-(target_win-1)], target_col)   
    main_ensemble = EnsemblePredictor(target_df, target_col)   

    # Ridge 계열 추가 -> embedding 된 data도 있으므로 괜찮을 듯    
    ku.add_ridge_based_pipe(main_ensemble, ts_embed)
    # kernel-ridge with RBF
    ku.add_kridge_based_pipe(main_ensemble, ts_embed)    
    # RandomForest 계열 추가
    ku.add_rf_based_pipe(main_ensemble, ts_embed)
    # SVR 계열 추가
    ku.add_svr_based_pipe(main_ensemble, ts_embed)
    # XGBoost 계열 추가
    ku.add_xgb_based_pipe(main_ensemble, ts_embed)    
    
    print('--> Build Ensemble(Simple Avg)')
    main_ensemble.build_ensemble(SimpleAvgEnsemble())
    
    print('---> Call fitting')
    fitted_model = main_ensemble.fit()
    fitted_y = main_ensemble.y_fitted_all
    
    return (fitted_model, fitted_y)
    
#### main
if __name__ == "__main__":

    rolling_win = 20 # 데이터 통계 window
    target_win = 5 # 예측치를 만들 total windown
    ts_embed = True # embed를 할 경우 rolling_win만큼 앞이 잘려나감.
    # data load
    data_df, target_col = load_data("20030101", "20220627", 
                                    rolling_win=rolling_win, 
                                    target_win=target_win, 
                                    embed_ts=ts_embed)
    fitted_ensemble, fitted_y =  fit_predictor(data_df, "20030101", "20220621",                                                 
                                                target_win=target_win, 
                                                target_col=target_col,
                                                ts_embed=ts_embed)    
        
    #[TODO] predict 함수 빼내기
    print('---> predict')
    predicted = fitted_ensemble.predict(data_df[-(target_win-1):])
    pred_all = fitted_ensemble.predict(data_df[:-(target_win-1)])

    # [TODO] Score 함수 구현
    # 성능 테스트 및 플로팅 고민해보기
    target_col = ['T_KOSPI2_RET_SUM', 'T_KOSPI2_RET_STD']
    r2 = fitted_ensemble.score(data_df[:-(target_win-1)], data_df[target_col][:-(target_win-1)])
    pipe_scores = fitted_ensemble.pipe_score(data_df[:-(target_win-1)], data_df[target_col][:-(target_win-1)])
    print(f'total r2 score : {r2}')
    print(f'individual scores : {pipe_scores}')
    

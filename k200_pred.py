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
from sklearn.ensemble import RandomForestRegressor

# data and utility
from datagen import load_k200_data
import k200_util as ku

import time

#import IPython as ipy

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
        orig_df, df = load_k200_data(start_dt, end_dt, rolling_win=rolling_win, save_file=True)
    
    return df

# target window만큼의 향후 return sum, real vol 계산
# KOSPI2_RET을 사용
def calc_target_val(df, window):
    target = 'KOSPI2_RET'
    mvdf = pd.DataFrame()
    #mvdf['T_' + target + "_AVG"] = df[target].rolling(window).mean()
    mvdf['T_' + target + "_SUM"] = df[target].rolling(window).sum()
    mvdf['T_' + target + "_STD"] = df[target].rolling(window).std()
    mvdf.dropna(inplace=True)
    
    # 예측치이므로 윈도우만큼 땡겨서 붙여줌
    df_index = df.index[:-(window-1)]
    mvdf_t = mvdf.set_index(df_index)
    # 예측치 아닌 부부은 정보로 취합
    df_index2 = df.index[(window-1):]
    mvdf_o = mvdf.set_index(df_index2)
    col = [ "_".join([target, str(window), "SUM"]), "_".join([target, str(window), "STD"])]
    mvdf_o.columns = col
    # 정보 포함
    ku._target_rel.extend(col)
    
    df = pd.concat([df, mvdf_o, mvdf_t], axis=1)
    return df
    

# data 읽어서 가져오기
def prepare_k200(start_dt, end_dt, rolling_win=20, embed=False):    
    # file version
    
    df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win, \
                        from_file=True, \
                        path="./data/20030101_20220627_k200ewm.csv")
       
    """
    df = load_data_k200(start_dt, end_dt, rolling_win=rolling_win, \
                        from_file=True, \
                        path="./data/20030101_20220627_k200.csv")
    """
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
    # kernel-ridge with RBF --> Too bad prediction
    #ku.add_kridge_based_pipe(main_ensemble, ts_embed)
    # GLM based -> taget negative 문제
    #ku.add_glm_based_pipe(main_ensemble, ts_embed)
    # RandomForest 계열 추가
    ku.add_rf_based_pipe(main_ensemble, ts_embed)
    # SVR 계열 추가
    ku.add_svr_based_pipe(main_ensemble, ts_embed)
    # XGBoost 계열 추가
    ku.add_xgb_based_pipe(main_ensemble, ts_embed)    
    
    print('--> Build Ensemble(Simple Avg)')
    main_ensemble.build_ensemble(SimpleAvgEnsemble())
    #print('--> Build Ensemble(Random Forest)')
    #main_ensemble.build_ensemble(RandomForestRegressor(min_samples_leaf=5))
    
    print('---> Call fitting')
    fitted_model = main_ensemble.fit()
    fitted_y = main_ensemble.y_fitted_all
    r2_list = main_ensemble.pipe_score(target_df, target_df[target_col])    
    
    return (fitted_model, fitted_y, r2_list)

# predict
def predict(data_df, model, target_dates):
    dates_d = [dt.strptime(x, "%Y%m%d") for x in target_dates]    
    y_pred = []
    y_pred_all = []
    
    print('---> Call predicting')
    for d in dates_d:
        yy = model.predict(data_df[data_df.index == d])        
        y_pred.append(yy[0]) # matrix 형태로 넘어옴
        y_pred_all.append(model.y_pred_all[0])
        
    return np.array(y_pred), np.array(y_pred_all)
    
#### main
if __name__ == "__main__":

    rolling_win = 20 # 데이터 통계 window
    target_win = 6 # 예측치를 만들 total windown
    ts_embed = True # embed를 할 경우 rolling_win만큼 앞이 잘려나감.
    # data load
    data_df, target_col = load_data("20030101", "20220627", 
                                    rolling_win=rolling_win, 
                                    target_win=target_win, 
                                    embed_ts=ts_embed)

    # Fit and Prediction test
    data_df = data_df.dropna()
    dates_arr = np.array(data_df.index)
    # 첫 fitting limit    
    idx = np.where(dates_arr >= np.datetime64('2022-06-01'))[0]

    cnt = 0
    # fit and predict
    fit_dts = []
    pred_dts = []
    preds = []
    pred_alls = []
    gnd_truth = []
    r2_vals = []
    
    for an_idx in idx:
        if an_idx >= len(data_df) - target_win:
            print("!!!Test 완료!!!")
            break

        if cnt % 10 == 0:
            start_t = time.time()

            fit_end_dt = data_df.index[an_idx]
            pred_dt = data_df.index[an_idx+target_win-1]
            
            fit_dt_str = dt.strftime(fit_end_dt, "%Y%m%d")
            pred_dt_str = dt.strftime(pred_dt, "%Y%m%d")

            print(f"{cnt+1} : fit until {fit_dt_str}, predict {pred_dt_str}")
            
            fitted_model, fitted_y, r2_list = fit_predictor(data_df, "20030101", fit_dt_str,                                                 
                                                        target_win=target_win, 
                                                        target_col=target_col,
                                                        ts_embed=ts_embed)
            pred, pred_all = predict(data_df, fitted_model, [pred_dt_str])

            end_t = time.time()
            print(f".... {end_t-start_t:.2f} secs elapsed -> pred : [{pred[0][0]:.4f} {pred[0][1]:.4f}]" + 
                  f" truth : [{data_df.loc[pred_dt][target_col]['T_KOSPI2_RET_SUM']:.4f} {data_df.loc[pred_dt][target_col]['T_KOSPI2_RET_STD']:.4f}]")

            fit_dts.append(fit_end_dt)
            pred_dts.append(pred_dt)
            preds.append(pred)
            pred_alls.append(pred_all)
            gnd_truth.append(data_df.loc[pred_dt][target_col])
            
            tmp_l = []
            for _, an_r2 in r2_list: # 일단 순서대로 들어온다고 가정하긴 함
                tmp_l.append(an_r2)
            r2_vals.append(tmp_l)
        else:
            print(f"{cnt+1} : Skipping...")

        cnt += 1

    # dataframe으로 정리 후 CSV 또는 XLSX로 저장
    result_cols = ["FIT_END_DT", "PRED_DT", "TRUE_SUM", "TRUE_STD", "PRED_RET_SUM", "PRED_RET_STD"]
    n_outs = len(pred_alls[0][0])
    n_base = int(n_outs / len(target_col))

    for i in range(n_base):
        c1 = "_".join(["BASE", "SUM", str(i+1)])
        c2 = "_".join(["BASE", "STD", str(i+1)])        
        result_cols.append(c1)
        result_cols.append(c2)        
        
    for i in range(n_base):
        c3 = "_".join(["FIT_SCORE", str(i+1)])
        result_cols.append(c3)

    result_df = pd.DataFrame(columns=result_cols)
    for i, fit_dt, pred_dt, pred_ret, pred_base, true_y, an_r2l \
        in zip(range(len(fit_dts)), fit_dts, pred_dts, preds, pred_alls, gnd_truth, r2_vals):
        a_row = []
        a_row.append(fit_dt)
        a_row.append(pred_dt)
        a_row.append(true_y[target_col[0]])
        a_row.append(true_y[target_col[1]])
        a_row.extend(pred_ret[0])
        a_row.extend(pred_base[0])
        a_row.extend(an_r2l)
        result_df.loc[i] = a_row
    
    # save result to csv file
    file_name = dt.now().strftime("%Y-%m-%d_%H%M") + "_pred_result.csv"
    result_df.to_csv(file_name, index=False)
    
    # calc r2 score for predictions
    y_true = np.array(result_df[["TRUE_SUM", "TRUE_STD"]])
    y_hat = np.array(result_df[["PRED_RET_SUM", "PRED_RET_STD"]])
    
    u = ((y_true - y_hat)**2).sum()
    v = ((y_true - y_true.mean())**2).sum()
    r2 = 1. - u/v
    print(f" === Final r2 score of prediction : {r2:.4f}")
    
    # base learner별 prediction score
    r2b_list = []
    for i in range(n_base):
        colb = ["_".join(["BASE", "SUM", str(i+1)]), "_".join(["BASE", "STD", str(i+1)])]
        y_hatb = np.array(result_df[colb])
        
        ub = ((y_true - y_hatb)**2).sum()        
        r2b = 1. - ub/v
        r2b_list.append(r2b)
    
    # 출력
    print("--- Base learer prediction scores ---")
    for i, an_r2 in zip(range(n_base), r2b_list):
        print(f" - Base {i+1} : {an_r2:.4f}")
    
    """
        
    fitted_ensemble, fitted_y =  fit_predictor(data_df, "20030101", "20220620",                                                 
                                                target_win=target_win, 
                                                target_col=target_col,
                                                ts_embed=ts_embed)    
        
    # predict
    print('---> predict')
    #predicted = fitted_ensemble.predict(data_df[-(target_win-1):])
    predicted, predicted_all = predict(data_df, fitted_ensemble, ['20220621', '20220627'])
    pred_all = fitted_ensemble.predict(data_df[:-(target_win-1)])

    # [TODO] Score 함수 구현
    # 성능 테스트 및 플로팅 고민해보기
    target_col = ['T_KOSPI2_RET_SUM', 'T_KOSPI2_RET_STD']
    r2 = fitted_ensemble.score(data_df[:-(target_win-1)], data_df[target_col][:-(target_win-1)])
    pipe_scores = fitted_ensemble.pipe_score(data_df[:-(target_win-1)], data_df[target_col][:-(target_win-1)])
    print(f'total r2 score : {r2}')
    print(f'individual scores : {pipe_scores}')
    """    
    

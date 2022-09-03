# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:43:46 2022

@author: SE19078
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

def load_result():
    file = "./result/2022-08-28_1701_pred_result.csv"
    result_df = pd.read_csv(file, index_col="PRED_DT")
    return result_df

def load_data():
    file = "./data/20030101_20220627_k200ewm.csv"
    data_df = pd.read_csv(file, index_col="STD_DT2")
    return data_df

def load_raw_data():
    file = "./data/20030101_20220627_k200.csv"
    data_df = pd.read_csv(file, index_col="STD_DT2")
    return data_df

def plot_prediction(target_dt, result_df, data_df, raw_df, win_size):
    # result 찾기
    print(f"plotting {target_dt}")
    a_result = result_df.loc[target_dt]
    fit_dt = a_result["FIT_END_DT"]
    
    plot_df = pd.DataFrame()
    # 날짜 및 주가 리스트
    idx = np.where(np.array(data_df.index) == target_dt)[0][0]# single value
    d_slice = data_df.index[(idx-20):(idx+win_size)]
    
    plot_df["KOSPI2(ewm)"] = data_df.loc[d_slice]["KOSPI2"]
    plot_df = plot_df.set_index(d_slice)
    plot_df["KOSPI2"] = raw_df.loc[d_slice]["KOSPI2"]
    
    # ewm 주가 추정하기(prediction)
    p_sum = a_result["PRED_RET_SUM"] * 0.01
    p_std = a_result["PRED_RET_STD"] * 0.01
    r = data_df.loc[target_dt]["KOSPI2_RET"] * 0.01
    p_avg = 1.0 + (p_sum - r) / float(win_size - 1)    
    
    pstk = [data_df.loc[target_dt]["KOSPI2"]]
    pstk_u = [pstk[0]]
    pstk_d = [pstk[0]]
    
    for i in range(1, win_size):
        pstk.append(pstk[i-1] * p_avg)
        pstk_u.append(pstk[i] * (1.0 + p_std*i))
        pstk_d.append(pstk[i] * (1.0 - p_std*i))
    
    pplot_df = pd.DataFrame()
    pplot_df["PRED_IDX"] = pstk
    pplot_df["PRED_UPPER"] = pstk_u
    pplot_df["PRED_LOWER"] = pstk_d
    pplot_df = pplot_df.set_index(d_slice[-win_size:])
    
    # ewm 주가 추정하기(truth)
    t_sum = a_result["TRUE_SUM"] * 0.01
    t_std = a_result["TRUE_STD"] * 0.01
    r = data_df.loc[target_dt]["KOSPI2_RET"] * 0.01
    t_avg = 1.0 + (t_sum - r) / float(win_size - 1)    
    
    tstk = [data_df.loc[target_dt]["KOSPI2"]]
    tstk_u = [pstk[0]]
    tstk_d = [pstk[0]]
    
    for i in range(1, win_size):
        tstk.append(tstk[i-1] * t_avg)
        tstk_u.append(tstk[i] * (1.0 + t_std*i))
        tstk_d.append(tstk[i] * (1.0 - t_std*i))
        
    tplot_df = pd.DataFrame()
    tplot_df["TRUE_IDX"] = tstk
    tplot_df["TRUE_UPPER"] = tstk_u
    tplot_df["TRUE_LOWER"] = tstk_d
    tplot_df = tplot_df.set_index(d_slice[-win_size:])
    
    ## PLOT
    ## PREDICTION PLOT
    fig, ax = plt.subplots(figsize=(10, 8))
    if p_avg < 1.0:
        co = "navy"
    else:
        co = "darkred"
        
    ax.grid()
    ax.plot(plot_df.index, plot_df["KOSPI2(ewm)"], color="seagreen", ls="-", lw=3.0, label="KOSPI2(ewma)")
    ax.plot(plot_df.index, plot_df["KOSPI2"], color="C1", ls="--", lw=3.0, label="KOSPI2", alpha=0.3)
    ax.plot(pplot_df.index, pplot_df["PRED_IDX"], color=co, ls="-", lw=4.0, label="Predicted")
    ax.fill_between(pplot_df.index, pplot_df["PRED_UPPER"], pplot_df["PRED_LOWER"], color=co, alpha=0.2)
    ax.set_ylabel("KOSPI2")
    ax.set_xlabel("Dates")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_title("KOSPI200 Prediction", fontsize=20)
    ax.legend(loc="lower left", fontsize=14)
    ax.axvline(x=fit_dt, color="black", lw=2.0, ls="--")
    #fig.show()
    fig.savefig("./fig/" + target_dt + "_pred.png")
    
    # TRUTH
    fig, ax = plt.subplots(figsize=(10, 8))
    if t_avg < 1.0:
        co = "navy"
    else:
        co = "darkred"
        
    ax.grid()
    ax.plot(plot_df.index, plot_df["KOSPI2(ewm)"], color="seagreen", ls="-", lw=3.0, label="KOSPI2(ewma)")
    ax.plot(plot_df.index, plot_df["KOSPI2"], color="C1", ls="--", lw=3.0, label="KOSPI2", alpha=0.3)
    ax.plot(tplot_df.index, tplot_df["TRUE_IDX"], color=co, ls="-", lw=4.0, label="Projected")
    ax.fill_between(tplot_df.index, tplot_df["TRUE_UPPER"], tplot_df["TRUE_LOWER"], color=co, alpha=0.2)
    ax.set_ylabel("KOSPI2")
    ax.set_xlabel("Dates")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_title("KOSPI200 Truth", fontsize=20)
    ax.legend(loc="lower left", fontsize=14)
    ax.axvline(x=fit_dt, color="black", lw=2.0, ls="--")
    #fig.show()
    fig.savefig("./fig/" + target_dt + "_truth.png")
            
    pass
    

if __name__ == "__main__":
    result_df = load_result()
    data_df = load_data()
    raw_df = load_raw_data()
    
    # corr chart
    corr_targets = ["KOSPI2", "VKOSPI", "USDKRW", "CRUDE_F", "SPX", "ROKCDS"]
    feat_df = data_df[corr_targets]
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(feat_df.corr(), annot=True, fmt=".3f", ax=ax)
    plt.show()
    f.savefig("./fig/data_corr.png")
    
    # 방향 정확도
    pred_sum = np.array(result_df["PRED_RET_SUM"])
    true_sum = np.array(result_df["TRUE_SUM"])
    mult = pred_sum * true_sum
    n_right = len(mult[mult >= 0])
    acc = float(n_right) / float(len(mult)) * 100.0
    print(f"-- Direction accuracy : {acc:.2f}%, {n_right} of {len(mult)}")

    # 하락 정확도
    n_idx = np.where(true_sum <= 0.0)[0]
    n_pred = pred_sum[n_idx]
    n_true = true_sum[n_idx]
    mult2 = n_pred * n_true
    n_right = len(mult2[mult2 >= 0])
    acc = float(n_right) / float(len(mult2)) * 100.0
    print(f"-- Up direction accuracy : {acc:.2f}%, {n_right} of {len(mult2)}")

    # 상승 정확도
    p_idx = np.where(true_sum > 0.0)[0]
    p_pred = pred_sum[p_idx]
    p_true = true_sum[p_idx]
    mult3 = p_pred * p_true
    n_right = len(mult3[mult3 >= 0])
    acc = float(n_right) / float(len(mult3)) * 100.0
    print(f"-- Down direction accuracy : {acc:.2f}%, {n_right} of {len(mult3)}")
    
    # 가장 안 맞고 잘 맞는 곳
    min_idx = np.argmin(mult)
    max_idx = np.argmax(mult)
    worst_dt = result_df.index[min_idx]
    best_dt = result_df.index[max_idx]
    
    print(f"Worst date : {worst_dt}, Best date : {best_dt} ")
    
    sorted_m = mult.copy()
    sorted_m.sort()    
    
    # plot and save
    """
    for a_date in result_df.index:
        # plot_prediction(a_date, result_df, data_df, raw_df, 6)
        # break
        try:
            plot_prediction(a_date, result_df, data_df, raw_df, 6)
        except:
            print(f"err -> {a_date}")
            continue # 다음으로 진행
    """
    
    
    
    
    
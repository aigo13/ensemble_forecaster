# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:43:46 2022

@author: SE19078
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_result():
    file = "./result/2022-08-28_0046_pred_result.csv"
    result_df = pd.read_csv(file, index_col="PRED_DT")
    return result_df

def load_data():
    file = "./data/20030101_20220627_k200ewm.csv"
    data_df = pd.read_csv(file, index_col="STD_DT2")
    return data_df

if __name__ == "__main__":
    result_df = load_result()
    data_df = load_data()
    
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
    
    # 가장 안 맞고 잘 맞는 곳
    min_idx = np.argmin(mult)
    max_idx = np.argmax(mult)
    worst_dt = result_df.index[min_idx]
    best_dt = result_df.index[max_idx]
    
    print(f"Worst date : {worst_dt}, Best date : {best_dt} ")
    
    sorted_m = mult.copy()
    sorted_m.sort()    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Data generation에 도움을 줄 수 있는 함수, 클래스 모음.
Created on Wed Jul  6 09:14:58 2022

@author: SE19078
"""

import pandas as pd
import numpy as np

def embed_ts(df, target_col, n_k):
    """
    지정된 time serise를 n_k 갯수만큼 임베딩함.
    y_i를 k개 임베딩할 경우 [y_(i-k-1) .... y_i ] 가 됨

    Parameters
    ----------
    df : DataFrame
        모든 data를 가지고있는 dataframe
    target_col : string 
        Embedding을할 target coloumn
    n_k : integer
        embedding size

    Returns
    -------
    embedding data를 가지고 있는 data frame.
    문제가 있을경우 None을 return

    """    
        
    col_names= [target_col+"_em"+str(x+1) for x in range(n_k)]
    ret_df = pd.DataFrame(columns=col_names)
    
    a_slice = np.array(df[target_col])
    for i in range(len(a_slice)):
        if i+n_k > len(a_slice):
            break        
        a_row = a_slice[i:i+n_k]                
        ret_df.loc[len(ret_df)] = a_row
        
    return ret_df


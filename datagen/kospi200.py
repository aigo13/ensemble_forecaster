# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:34:31 2022

@author: SE19078
"""

import pandas as pd
from . import DBhandle as db
#import DBhandle as db
import numpy as np
from datetime import datetime as dt
#import matplotlib.pyplot as plt


def LoadInner(sql, convert_dt, lagging=0):
    # DB에서 가져오기
    conn = db.ConnectDB()
    df = db.LoadQuery(sql, conn)
    db.CloseDB(conn)
    
    # convert_dt가 TRUE이면 
    if convert_dt == True:        
        str_arr = np.array(df['STD_DT'])        
        dt_arr = [dt.strptime(x, '%Y%m%d') for x in str_arr]        
        df['STD_DT2'] = dt_arr
        
        pdf = df.pivot(index='STD_DT2', columns='CODE', values='CLS_PRC')            
    else:
        pdf = df.pivot(index='STD_DT', columns='CODE', values='CLS_PRC')
        
    # lagging이 지정됐을 경우 처리
    if lagging > 0:
        for a_col in pdf.columns:
            val_arr = np.array(pdf[a_col])
            val_arr = val_arr[:-lagging]
            beg_v = val_arr[:lagging]
            val_arr = np.insert(val_arr, 0, beg_v)
            pdf[a_col] = val_arr
    elif lagging < 0:
        for a_col in pdf.columns:
            val_arr = np.array(pdf[a_col])
            val_arr = val_arr[-lagging:]
            end_v = val_arr[lagging:]
            val_arr = np.insert(val_arr, len(val_arr), end_v)
            pdf[a_col] = val_arr
        
    return pdf

"""
    Load KOSPI200 and VKOSPI data from DB(IVG01B)
"""
def LoadIndexDB(start_str, end_str, convert_dt=False, dropna=False):
    sql = """select STD_DT, DECODE(TRIM(HF_BLMGRG), 'KOSPI2 INDEX', 'KOSPI2', 'VKOSPI INDEX', 'VKOSPI') as CODE, 
             THDY_CLS_PRC as CLS_PRC from IVG01B 
             where TRIM(HF_BLMGRG) in ('KOSPI2 INDEX', 'VKOSPI INDEX')
             and STD_DT >= ':ST_DT' and STD_DT <= ':END_DT'"""
    sql = sql.replace(':ST_DT', start_str)
    sql = sql.replace(':END_DT', end_str)
    pdf = LoadInner(sql, convert_dt=convert_dt)
    
    if dropna == True:
        pdf.dropna(subset=['KOSPI2'], inplace=True)
        pdf.interpolate(method='linear', inplace=True)
        
    return pdf

# 환율, 환율 변동성 Comodity 가격(CL1) 불러오기
def LoadFXDB(start_str, end_str, convert_dt=False, dropna=False):
    sql = """select STD_DT, DECODE(TRIM(HF_BLMGRG), 'USDKRW CURNCY', 'USDKRW', 
              'USDKRWV1M CURNCY', 'USDKRW_V') as CODE, 
             THDY_CLS_PRC as CLS_PRC from IVG01B 
             where TRIM(HF_BLMGRG) in ('USDKRW CURNCY', 'USDKRWV1M CURNCY')
             and STD_DT >= ':ST_DT' and STD_DT <= ':END_DT'"""
    
    sql = sql.replace(':ST_DT', start_str)
    sql = sql.replace(':END_DT', end_str)
    pdf = LoadInner(sql, convert_dt=convert_dt)
    
    if dropna == True:
        # USDKRW 기준으로 drop하고 Crude oil은 interpolation으로 filling
        pdf.dropna(subset=['USDKRW'], inplace=True)
        pdf.interpolate(method='linear', inplace=True)
        
    return pdf


# 환율, 환율 변동성 Comodity 가격(CL1) 불러오기 - 하루 lagging함(NY)
def LoadCrudeDB(start_str, end_str, convert_dt=False, dropna=False):
    sql = """select STD_DT, DECODE(TRIM(HF_BLMGRG), 'CL1 COMDTY', 'CRUDE_F' ) as CODE, 
             THDY_CLS_PRC as CLS_PRC from IVG01B 
             where TRIM(HF_BLMGRG) in ('CL1 COMDTY')
             and STD_DT >= ':ST_DT' and STD_DT <= ':END_DT'"""
    
    sql = sql.replace(':ST_DT', start_str)
    sql = sql.replace(':END_DT', end_str)
    pdf = LoadInner(sql, convert_dt=convert_dt, lagging=1) # 전일치로 하루 밀어줌(전일 값->당일 kospi200)
    
    if dropna == True:
        # USDKRW 기준으로 drop하고 Crude oil은 interpolation으로 filling
        pdf.dropna(subset=['CRUDE_F'], inplace=True)        
        
    return pdf

#  SPX Vol, VIX 불러오기 - 하루 lagging함(NY)
def LoadSpxDB(start_str, end_str, convert_dt=False, dropna=False):
    sql = """select STD_DT, DECODE(TRIM(HF_BLMGRG), 'SPX INDEX', 'SPX', 'VIX INDEX', 'VSPX' ) as CODE, 
             THDY_CLS_PRC as CLS_PRC from IVG01B 
             where TRIM(HF_BLMGRG) in ('SPX INDEX', 'VIX INDEX')
             and STD_DT >= ':ST_DT' and STD_DT <= ':END_DT'"""
    
    sql = sql.replace(':ST_DT', start_str)
    sql = sql.replace(':END_DT', end_str)
    pdf = LoadInner(sql, convert_dt=convert_dt, lagging=1) # 전일치로 하루 밀어줌(전일 값->당일 kospi200)
    
    if dropna == True:        
        pdf.dropna(subset=['SPX'], inplace=True)        
        
    return pdf

#  SPX Vol, VIX 불러오기 - 하루 lagging함(NY)
def LoadCdsDB(start_str, end_str, convert_dt=False, dropna=False):
    sql = """select STD_DT, DECODE(TRIM(HF_BLMGRG), 'CKREA1U5 CURNCY', 'ROKCDS') as CODE, 
             (BIDPRC + ASKPRC)/2 as CLS_PRC from IVG01B 
             where TRIM(HF_BLMGRG) in ('CKREA1U5 CURNCY')
             and STD_DT >= ':ST_DT' and STD_DT <= ':END_DT'"""
    
    sql = sql.replace(':ST_DT', start_str)
    sql = sql.replace(':END_DT', end_str)
    pdf = LoadInner(sql, convert_dt=convert_dt, lagging=0) # 전일치로 하루 밀어줌(전일 값->당일 kospi200)
    
    if dropna == True:        
        pdf.dropna(subset=['ROKCDS'], inplace=True)        
        
    return pdf


# 금리 스프레드를 구하기 위한 쿼리
def LoadIRSpread(start_str, end_str, convert_dt=False):
    sql = """select STD_DT, DECPDE(ITRST_ID, '1013000', 'KTB_3Y', '6013123', 'FINAA_3Y') as CODE, YLD
             from RRP00B@LINK_HDRVD_I_P where ITRST_ID in( '1013000', '6013123') and ITRST_CCD= '1' 
             AND RMD_MTRT_CD ='M036' and STD_DT >= ':ST_DT' and STD_DT <= ':END_DT'"""
    sql = sql.replace(':ST_DT', start_str)
    sql = sql.replace(':END_DT', end_str)
    pdf = LoadInner(sql, convert_dt=convert_dt)    
    pdf['AA_KTB_SPRD_3Y'] = pdf['FINAA_3Y'] - pdf['KTB_3Y']
    
    return pdf


# 실현 변동성 구하기. Log Returen 기준
def LogRet(prc_arr):
    yy0 = np.array(prc_arr[:-1])
    yy1 = np.array(prc_arr[1:])
    ret = np.log(yy1) - np.log(yy0) # log return    
    return ret


# 지정한 column의 stat 구하기
def AddStats(df, col_name, annualize_std = False, window=30, year_d=365):
    mfactor = 1.0
    if annualize_std == True:
        mfactor = np.sqrt(year_d)
    
    ndf = df.dropna(subset=[col_name])    
    
    df[col_name + '_AVG'] = ndf[col_name].rolling(window).mean()
    df[col_name + '_MIN'] = ndf[col_name].rolling(window).min()
    df[col_name + '_MAX'] = ndf[col_name].rolling(window).max()
    df[col_name + '_AMP'] = df[col_name + '_MAX'] - df[col_name + '_MIN']
    df[col_name + '_STD'] = ndf[col_name].rolling(window).std() * mfactor
    df[col_name + '_SKEW'] = ndf[col_name].rolling(window).skew()    
    df[col_name + '_SUM'] = ndf[col_name].rolling(window).sum()


def load_k200_data(start_str, end_str, rolling_win=20, save_file=False):
    """
    지정된 시작일부터 종료일 까지의 KOSPI200과 관련 데이터 로드
    지수, VIX, Log수익률, 실현 변동성, WTI선물가격, 원달러환율, 원달러환율 변동성
    지정된 시작일+rolling_win일부터 데이터 제공

    Parameters
    ----------
    start_str : String
        %Y%m%d 형식의 날짜 string. 데이터 시작일
    end_str : String
        %Y%m%d 형식의 날짜 string. 데이터 완료일
    rolling_win : Integer
        실현 변동성 등을 구하기 위한 시간 window
        
    Returns
    -------
    (Dataframe, Dataframe) 원본 데이터를 담고있는 dataframe과 EWM 스무딩된 데이터의 dataframe

    """
    df = LoadIndexDB(start_str, end_str, convert_dt=True, dropna=True)    
    df2 = LoadFXDB(start_str, end_str, convert_dt=True, dropna=True)
    df3 = LoadCrudeDB(start_str, end_str, convert_dt=True, dropna=True)
    df4 = LoadSpxDB(start_str, end_str, convert_dt=True, dropna=True)
    df5 = LoadCdsDB(start_str, end_str, convert_dt=True, dropna=True)
    
    df = pd.concat([df, df2, df3, df4, df5], axis=1)    
    # KOSPI200 기준으로 클린즈, 그 외는 interpolation
    df.dropna(subset=['KOSPI2'], inplace=True)
    df.interpolate(method='linear', inplace=True)  
    
    # ewm으로 스무딩한 값의 return
    ewm_df = df.ewm(alpha=0.3, adjust=False).mean()
    rv_ret = LogRet(ewm_df['KOSPI2'])    
    ret_df = pd.DataFrame()
    ret_df['KOSPI2_RET'] = rv_ret * 100.0
    ret_df = ret_df.set_index(ewm_df.index[1:]) 
    
    fx_ret = LogRet(ewm_df['USDKRW'])    
    ret2_df = pd.DataFrame()
    ret2_df['FX_RET'] = fx_ret * 100.0    
    ret2_df = ret2_df.set_index(ewm_df.index[1:]) 
    
    crude_ret = LogRet(ewm_df['CRUDE_F'])
    ret3_df = pd.DataFrame()
    ret3_df['CRUDE_RET'] = crude_ret * 100.0    
    ret3_df = ret3_df.set_index(ewm_df.index[1:])
    
    spx_ret = LogRet(ewm_df['SPX'])
    ret4_df = pd.DataFrame()
    ret4_df['SPX_RET'] = spx_ret * 100.0    
    ret4_df = ret4_df.set_index(ewm_df.index[1:])
    
    ewm_df = pd.concat([ewm_df, ret_df, ret2_df, ret3_df, ret4_df], axis=1)
    ewm_df.dropna(inplace=True)

    # RAW data로 생성한 return
    # return 통계치
    rv_ret = LogRet(df['KOSPI2'])    
    ret_df = pd.DataFrame()
    ret_df['KOSPI2_RET'] = rv_ret * 100.0
    ret_df = ret_df.set_index(df.index[1:])
    
    # FX Log return
    fx_ret = LogRet(df['USDKRW'])    
    ret2_df = pd.DataFrame()
    ret2_df['FX_RET'] = fx_ret * 100.0    
    ret2_df = ret2_df.set_index(df.index[1:])        
    
    # WTI Log return
    crude_ret = LogRet(df['CRUDE_F'])    
    ret3_df = pd.DataFrame()
    ret3_df['CRUDE_RET'] = crude_ret * 100.0    
    ret3_df = ret3_df.set_index(df.index[1:])  

    # SPX Log return
    spx_ret = LogRet(df['SPX'])
    ret4_df = pd.DataFrame()
    ret4_df['SPX_RET'] = spx_ret * 100.0    
    ret4_df = ret4_df.set_index(df.index[1:])      
           
    # df에 합침    
    df = pd.concat([df, ret_df, ret2_df,ret3_df, ret4_df], axis=1)    
    df.dropna(inplace=True)    
        
    # ewma statistics
    AddStats(ewm_df, 'KOSPI2_RET', annualize_std=True, window=rolling_win) 
    AddStats(ewm_df, 'FX_RET', annualize_std=True, window=rolling_win)    
    AddStats(ewm_df, 'CRUDE_RET', annualize_std=True, window=rolling_win)
    AddStats(ewm_df, 'SPX_RET', annualize_std=True, window=rolling_win)
    ewm_df['VSPREAD'] =  ewm_df['KOSPI2_RET_STD'] - ewm_df['VKOSPI']    
    ewm_df['VIX_SPREAD'] =  ewm_df['SPX_RET_STD'] - ewm_df['VSPX']
    
    # add statistics
    AddStats(df, 'KOSPI2_RET', annualize_std=True, window=rolling_win)
    AddStats(df, 'FX_RET', annualize_std=True, window=rolling_win)
    AddStats(df, 'CRUDE_RET', annualize_std=True, window=rolling_win)
    AddStats(df, 'SPX_RET', annualize_std=True, window=rolling_win)
    df['VSPREAD'] =  df['KOSPI2_RET_STD'] - df['VKOSPI']
    df['VIX_SPREAD'] =  df['SPX_RET_STD'] - df['VSPX']
    
    # Final Cleanse
#    df.dropna(inplace=True)
#    ewm_df.dropna(inplace=True)
    
    if save_file == True:
        file_nm = [start_str, end_str, 'k200.csv']
        efile_nm = [start_str, end_str, 'k200ewm.csv']
        dfile = '_'.join(file_nm)
        efile = '_'.join(efile_nm)
        df.to_csv('../data/' + dfile)
        ewm_df.to_csv('../data/' + efile)
        
#        df.to_parquet('../data/' + dfile, compression='gzip')
#        df.to_parquet('../data/' + efile, compression='gzip')
    
    return (df, ewm_df)


# if __name__ == "__main__":
#     start_str = '20030101'
#     end_str = '20220627'    
    
#     df, ewm_df = LoadK200Data(start_str, end_str, rolling_win=20, save_file=True)
   
    
    # df['FVSPREAD'] = df['K200_FRV'] - df['VKOSPI']    
    
    # return rolling mean
    # df['SUM_RET'] = df['K200_RET'].rolling(window).sum() * 100
    
    # fwdret_df = pd.DataFrame()
    # fwdret_df['FSUM_RET'] = np.array(df['SUM_RET'].dropna(axis=0))
    # fwdret_df = fwdret_df.set_index(df.index[1:-1*(window-1)])
    
    # df = pd.concat([df, fwdret_df], axis=1)    
    
#    fig, ax = plt.subplots() #figsize=(10,10))
#    #ax.plot(df.index[-530:-30], rv[-500:], df.index[-530:-30], np.array(df['VKOSPI'])[-530:-30])
#    ax.plot(df.index[-500:], rv[-500:], df.index[-500:], np.array(df['VKOSPI'])[-500:])
#    ax2 = ax.twinx()
#    ax2.plot( df.index[-500:], np.array(df['KOSPI2'])[-500:], color='red')
#    plt.show()
#    
#    # Imvol vs Real Vol
#    vdiff =  np.array(df['VKOSPI'])[-500:] - rv[-500:]
#    fig, ax = plt.subplots()
#    ax.plot(df.index[-500:], np.array(df['KOSPI2'])[-500:])
#    ax2 = ax.twinx()
#    ax2.plot( df.index[-500:], vdiff, color='red')
#    plt.show()

#    # # Smoothed plot
    # fig, ax = plt.subplots()
    # ax.plot(ewm_df.index[-500:], np.array(ewm_df['KOSPI2'])[-500:])
    # ax2 = ax.twinx()
    # ax2.plot(ewm_df.index[-500:], np.array(ewm_df['USDKRW'])[-500:], color='red')
    #  #ax2.plot(ewm_df.index[-200:], np.array(df['VSPREAD'])[-200:], color='black')
    # plt.show()
#    
    # # -> 향후 SUM_RET, VSPREAD를 타겟으로 하는
    
#    c19_start = dt.strptime('20200101', '%Y%m%d')
#    c19_end = dt.strptime('20210101', '%Y%m%d')
#    ewm_slice = ewm_df[(ewm_df.index >= c19_start) & (ewm_df.index <= c19_end)]
#    df_slice = df[(ewm_df.index >= c19_start) & (ewm_df.index <= c19_end)]
#    fig, ax = plt.subplots()
#    #ax.plot(ewm_slice.index, np.array(ewm_slice['KOSPI2']))
#    ax.plot(ewm_slice.index, np.array(ewm_slice['VSPREAD']))
#    ax2 = ax.twinx()
#    #ax2.plot(ewm_slice.index, np.array(ewm_slice['VSPREAD']), color='red')
#    #ax2.plot(ewm_slice.index, np.array(ewm_slice['K200_RET']), color='red')
#    ax2.plot(ewm_slice.index, np.array(ewm_slice['SUM_RET']), color='red')
#    #ax2.plot(ewm_df.index[-200:], np.array(df['VSPREAD'])[-200:], color='black')
#    plt.show()
    
 # fwdrv_df = pd.DataFrame()
 # fwdrv_df['K200_FRV'] = rv_ret[0]
 # fwdrv_df = fwdrv_df.set_index(df.index[1:-1*(window-1)])    
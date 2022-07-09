# -*- coding: utf-8 -*-
"""
statsmodel의 ARIMAX를 이용한 base learner

@author: SE19078
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np
from datetime import datetime as dt

"""
    ArimaX learner 모델
"""
class ArimaX():
    
    """
        df : data frame(index : date)
        y_nm : target name
        exo_nm : exogeneous variable name        
        q, d, q : for ARIMA model parameters
    """
    def __init__(self, df, y_nm, exo_nm, order=(1,0,1), exo_order=(1,0,1)):
        self.data = df.copy()
        self.y_nm = y_nm
        self.exo_nm = exo_nm
        self.order = order
        self.exo_order = exo_order
        
        if exo_nm is not None:
            # target 추정용 모델
            self.endo_model = sm.tsa.statespace.SARIMAX(self.data[self.y_nm], 
                                                     exog=self.data[self.exo_nm], 
                                                     order=self.order)
            # 외삽 변수 추정용 모델
            self.exo_model = sm.tsa.statespace.SARIMAX(self.data[self.exo_nm],                                                      
                                                     order=self.exo_order)
        else:
            # target 추정용 모델
            self.endo_model = sm.tsa.statespace.SARIMAX(self.data[self.y_nm], 
                                                     order=self.order)
            self.exo_model = None
                
    """
        fit data
    """
    def fit(self):
        fit_res = self.endo_model.fit(disp=False)        
        self.res = self.endo_model.filter(fit_res.params)
        
        if self.exo_model is not None:
            fit_exo = self.exo_model.fit(disp=False)
            self.exo_res = self.exo_model.filter(fit_exo.params)
        return self.res
        
    """
        predict steps-ahead value
    """
    def predict(self, start_dt, end_dt):
        if self.exo_model is not None:
            exo_pred =  self.exo_res.get_prediction(start=start_dt, end=end_dt)        
            pred = self.res.get_prediction(start=start_dt, end=end_dt, exog=exo_pred)
        else:
            pred = self.res.get_prediction(start=start_dt, end=end_dt)
            
        return pred
        

#### Test

if __name__ == "__main__":
    import DBhandle as db
    import loess as ls
    
    # KOSPI200, VKOSPI 불러오기
    conn = db.ConnectDB()
    sql = """select STD_DT, TRIM(HF_BLMGRG) as CODE, THDY_CLS_PRC as CLS_PRC 
          from IVG01B where TRIM(HF_BLMGRG) in ('KOSPI2 INDEX', 'VKOSPI INDEX') 
          and STD_DT >= '20030102' order by STD_DT"""
    df = db.LoadQuery(sql, conn)
    db.CloseDB(conn)
    
    dt_str = np.array(df['STD_DT'])
    dt_dt = [dt.strptime(str(x), "%Y%m%d") for x in dt_str ]
    df['DT'] = dt_dt
    
    pdf = df.pivot(index='DT', columns='CODE', values='CLS_PRC')
    
    # smoothing 적용을 위하여 epoch_d로 만듦
    dt_arr = pdf.index
    base_dt = dt(1970, 1,1)
    epoch_d = [(x-base_dt).days for x in dt_arr]
    epoch_d = np.array(epoch_d)
    
    # Smoothing
    lss = ls.Loess(epoch_d, np.array(pdf['KOSPI2 INDEX']))
    smoothed = np.zeros_like(np.array(pdf['KOSPI2 INDEX']))
    yi = 0
    for an_x in epoch_d:
        smoothed[yi] = lss.Estimate(an_x, window=5) # 5영업일 smothing 느낌...
        yi += 1
        
    pdf['KOSPI2_LS'] = smoothed
    
    lss2 = ls.Loess(epoch_d, np.array(pdf['VKOSPI INDEX']))
    smoothed2 = np.zeros_like(np.array(pdf['VKOSPI INDEX']))
    yi = 0
    for an_x in epoch_d:
        smoothed2[yi] = lss2.Estimate(an_x, window=5)
        yi += 1
    pdf['VKOSPI_LS'] = smoothed2
    
    
    
    

 
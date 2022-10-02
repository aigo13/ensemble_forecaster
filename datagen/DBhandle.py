# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:02:22 2022

@author: SE19078
"""

import cx_Oracle as ora
import pandas as pd

'''
    connect and close db
'''
def ConnectDB():
    test_dsn = ora.makedsn('XX.X.X.XX', 1234, 'XXXXXX')
    conn = ora.connect('#####', '#######', test_dsn)
    return conn

def CloseDB(conn):
    conn.close()


'''
    Load data with query
'''
def LoadQuery(query_str, conn, col=None):
    #import QMR_pyUtils as qutils
    
    #conn = qutils.test_db_connect()
    qr_str = query_str  
    df = pd.read_sql_query(qr_str, conn)
    
    #conn.close()  
    
    if col != None:
        res = df[col]
    else:
        res = df
        
    return res

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 08:42:28 2022

@author: SE19078
"""

import numpy as np
import time
import math


# tricubic weight 
def Tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1) # 거리가 -`~1 사이만
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y

# Loess Class
class Loess(object):
    
    # normalize -> max-min 거리로 정규화
    @staticmethod
    def NormalizeArray(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val
    
    # intiialize
    def __init__(self, xx, yy, degree=-1):
        self.n_xx, self.min_xx, self.max_xx = self.NormalizeArray(xx)
        self.n_yy, self.min_yy, self.max_yy = self.NormalizeArray(yy)
        self.degree = degree
    
    # minimal-distance window의 index를 갖는 indexing arary
    @staticmethod
    def GetMinRange(distances, window):
        min_idx = np.argmin(distances) # 제일 작은 수의 첫 번째 index
        n = len(distances)
        
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)
        
        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        
        return np.array(min_range)
    
    # 가중치 구하기
    @staticmethod
    def GetWeights(distances, min_range):
        max_distance = np.max(distances[min_range])        
        weights = Tricubic(distances[min_range] / max_distance)
        return weights
    
    # normalizae
    def NormalizeX(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)
    
    # denormalize
    def DenormalizeY(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy
    
    # Estimate
    # Loess로 스무딩한 결과를 얻어내는 함수
    def Estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.NormalizeX(x)
        distances = np.abs(self.n_xx - n_x)        
        min_range = self.GetMinRange(distances, window)
        weights = self.GetWeights(distances, min_range)        
        
        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree+1))
            
            xp = np.array([[math.pow(n_x, p)] for p in range(degree+1)])
            for i in range(1, degree+1):
                xm[:, i] = np.power(self.n_xx[min_range], i)
            
            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ wm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)
            
            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight
            
            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_y * sum_weight)
            a = mean_y - b* mean_x            
            y = a + b * n_x
        
        return self.DenormalizeY(y)
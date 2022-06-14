# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:22:18 2021

@author: wpl
"""

import matplotlib.pyplot as plt
import numpy as np
from date_function import *
from scipy.optimize import curve_fit
from graphplot import *
import math



'''输入array，输出相关系数并画出图像'''
class constituticveResult():
    def __init__(self,test,pre):
        self.test=test
        self.pre=pre
        self.correCoe=self.calCorreCoe()

        # relevance(self.test,self.pre,300)
        self.show_accurate()
        self.plot()
        
    def calCorreCoe(self):
        test=self.test.reshape(-1,1)
        pre=self.pre.reshape(-1,1)
        test_mean=np.mean(test)
        pre_mean=np.mean(pre)
        test_part=0
        pre_part=0
        bi_part=0
        for i in np.arange(len(self.test)):
            test_part=test_part+(test[i]-test_mean)**2
            pre_part=pre_part+(pre[i]-pre_mean)**2
            bi_part=bi_part+(test[i]-test_mean)*(pre[i]-pre_mean)
           
        test_part=test_part**0.5
        pre_part=pre_part**0.5
        correCoe=bi_part/(test_part*pre_part)
        print(test_part,pre_part,correCoe,bi_part)
        return correCoe
    
    def plot(self):
        test=self.test.reshape(-1,1)
        pre=self.pre.reshape(-1,1)
        fig=plt.figure()
        plt.plot(test,test,label='best line',color='r')
        plt.scatter(test,pre,label='pre',color='b')
        plt.xlabel('expirimental value')
        plt.ylabel('predicted value')
        plt.legend()
        plt.show()
        
        
    def show_accurate(self):
        num=0
        AARE=0
        RMSE=0
        test=self.test.reshape(-1,1)
        pre=self.pre.reshape(-1,1)
        for i in np.arange(len(test)):
            AARE+=abs((test[i]-pre[i])/test[i])
            RMSE+=(test[i]-pre[i])**2
            if (abs(test[i]-pre[i])/test[i])<0.1:

                num+=1
        num=num/len(test)
        AARE=AARE/len(test)
        RMSE=(RMSE/len(test))**0.5
        print("num=",num)
        print("AARE=",AARE)
        print("RMSE=",RMSE)
        print('corre',self.correCoe)
        
        
        
        
        
        
        
    

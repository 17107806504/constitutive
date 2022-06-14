# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:12:27 2021

@author: wpl
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from graphplot import *
import math
import result_show
from sklearn import linear_model
from scipy.optimize import curve_fit
import data
import result_show







class ZA_model():
    def __init__(self,yieldstress,ratedate):
        self.yieldstress=yieldstress
        self.ratedate=np.array(ratedate)
        # self.tempdate=tempdate
        self.rate_para()
    def rate_para(self):
        rate,strain,stress=self.ratedate
        rate=rate.reshape(-1)
        strain=strain.reshape(-1)
        stress=stress.reshape(-1)
        def func(X,a, b, n,c,d):
            x,y=X
            return a+b*y**n*np.exp(c+d*np.log(x))

        x=rate
        y=strain
        z=stress
        popt, pcov=curve_fit(func,(x,y), z,maxfev=500000)
        a, b, n,c,d= popt
        print(popt)
        rate,strain,stress=self.ratedate
        pre=func((rate,strain),a, b, n,c,d)
        result=result_show.constituticveResult(stress,pre)
        rate, strain, stress = self.ratedate
        pre = func((rate, strain), a, b, n,c,d)
        autoPlot_allInOne(strain, stress, strain, pre, [0.001, 0.01, 0.1, 1000, 2000, 3000])





strain,rate,temp,stress=data.date_diff_rate()
a=ZA_model(112,[rate,strain,stress])













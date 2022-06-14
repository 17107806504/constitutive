# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:03:36 2021

@author: wpl
"""

import matplotlib.pyplot as plt
import numpy as np
import data
from scipy.optimize import curve_fit
from graphplot import *
import math
import result_show
from sklearn import linear_model
from scipy.optimize import curve_fit

strain,rate,temp,stress=data.date_diff_rate()
staticRate=rate[0]
staticStrain=strain[0]

class Tm(object):
    def __init__(self,static,lowdate,highdate):
        self.staticStrain,self.staticStress,self.staticRate=np.array(static)
        self.low_strain,self.low_stress,self.low_rate=np.array(lowdate)
        self.high_strain, self.high_stress, self.high_rate = np.array(highdate)
        self.a,self.m1,self.beta=self.cal_para_c()
        self.B,self.m2=self.cal_B_m2()
        Tm.result([self.a,self.m1,self.beta,self.B,self.m2],static,[lowdate[i]+highdate[i] for i in np.arange(len(lowdate))])




    @staticmethod
    def get_first_part(a, m, b,strain,rate,staticStress):
        return (np.array(strain) ** m * a + b)*(1 - np.array(staticStress) / 220)*np.log(rate/0.001)+staticStress

    def cal_para_c(self):
        strain=[]
        stress=[]
        C=[]
        for i in np.arange(len(self.low_strain)):
            y = np.interp(self.staticStrain, self.low_strain[i], self.low_stress[i])
            stress.append(y)
            c = (((y - self.staticStress) / self.staticStress) / np.log(self.low_rate[i][0] / 0.001))/(1 / np.array(self.staticStress) - 1 / 220)
            strain.append(self.staticStrain)
            C.append(c)

        y =np.array(C).reshape(-1)

        def func(x, a, m, b):
            return a * x ** m + b

        x = np.array(strain).reshape(-1)

        popt, pcov = curve_fit(func, x, y, maxfev=500000)

        a, m, b = popt
        print(a, b)
        y_pre = Tm.get_first_part(a, m, b,strain[1],self.low_rate[1][0],self.staticStress)
        plt.plot(strain[1],y_pre)
        plt.plot(strain[0], stress[1])
        plt.show()
        return a,m,b

    def cal_B_m2(self):
        strain = []
        second_part = []
        rate=[]
        for i in np.arange(len(self.high_strain)):
            y=np.interp(self.staticStrain, self.high_strain[i], self.high_stress[i])
            y=y-Tm.get_first_part(self.a, self.m1, self.beta,self.staticStrain,self.high_rate[i][0],self.staticStress)
            second_part.append(y)
            strain.append(self.staticStrain)
        y=np.array(second_part).reshape(-1)
        x=np.log(self.high_rate.reshape(-1)/0.001)
        def func(x, B, m):
            return B * x ** m

        popt, pcov = curve_fit(func, x, y, maxfev=500000)
        B, m2=popt
        print(B, m2)
        return B,m2

    @staticmethod
    def result(para,static,diff_rate_data):
        a,m1,beta,B,m2=para
        static_strain,static_stress,staticrate=static
        diff_rate_strain,diff_rate_stress,diff_rate=diff_rate_data
        strain=[]
        stress=[]
        for i in np.arange(len(diff_rate)):
            strain.append( static_strain)
            y = np.interp(static_strain, diff_rate_strain[i], diff_rate_stress[i])
            stress.append(y)
        stress=np.array(stress)
        strain=np.array(strain)
        diff_rate=np.array(diff_rate)
        pre=Tm.get_first_part(a, m1, beta,strain,diff_rate,static_stress)+B*np.log(diff_rate/0.001)**m2
        autoPlot_allInOne(strain, stress, strain, pre, [ 0.01, 0.1, 1000, 2000, 3000])
        evalve = result_show.constituticveResult(stress, pre)




static=[strain[0],stress[0],rate[0]]
lowdate=[strain[1:3],stress[1:3],rate[1:3]]
highdate=[strain[3:],stress[3:],rate[3:]]
test=Tm(static,lowdate,highdate)
        
       
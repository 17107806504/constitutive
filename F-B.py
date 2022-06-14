import matplotlib.pyplot as plt
import numpy as np
import data
from scipy.optimize import curve_fit
from graphplot import *
import math
import result_show
from sklearn import linear_model
from scipy.optimize import curve_fit

strain,rate,temp,stress=data.date_diff_condition()
class Field_Backofen(object):
    def __init__(self,diff_date):
        print(np.array(diff_date))
        self.strain,self.rate,self.temp, self.stress=diff_date
        C,n,m,p,q=self.cal_para()
        # Field_Backofen.result([a, m1, b, B, m2], static,
        #           diff_date)
        Field_Backofen.result([C,n,m,p,q], diff_date)

    def cal_para(self):
        strain=np.array(self.strain).flatten()
        rate=np.array(self.rate).reshape(-1)
        T=np.array(self.temp).ravel()
        stress= np.array(self.stress).ravel()
        y=np.log(stress)


        x1=np.log(strain)
        print(rate)
        x2=np.log(rate)
        x3=T


        def func(X, lnC,n,m,p,q):
            x1,x2,x3=X
            return lnC+n*x1+m*x2+p*x3+q*np.exp(x1)



        popt, pcov = curve_fit(func, (x1,x2,x3), y, maxfev=500000)

        lnC,n,m,p,q = popt
        C=np.exp(lnC)
        print(C,n,m,p,q)
        return C,n,m,p,q

    @staticmethod
    def cal_stress( para,strain,rate,temp):
        C, n, m, p, q = para
        return C*(strain**n)*(rate**m)*np.exp(p*temp+q*strain)
    @staticmethod
    def result(para,  diff_date):


        strain, rate,temp,stress = np.array(diff_date)


        pre = Field_Backofen.cal_stress(para,strain,rate,temp)
        print(pre.shape)
        autoPlot_allInOne(strain, stress, strain, pre, [0.001,0.01, 0.1, 1000, 2000, 3000])
        evalve = result_show.constituticveResult(stress, pre)
test=Field_Backofen(data.date_diff_condition())

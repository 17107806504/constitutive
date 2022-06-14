import matplotlib.pyplot as plt
import numpy as np
from date_function import *
from scipy.optimize import curve_fit
from graphplot import *
import math
import result_show
from sklearn import linear_model
from scipy.optimize import curve_fit
import result_show
import data





#确定出n，b
class CS_model():
    def __init__(self,staticStrain,staticStress,yieldstress,tempdata,ratedate):
        self.staticStrain=staticStrain
        self.staticStress=staticStress
        self.yieldstress=yieldstress
        self.tempdata=tempdata
        # self.ratedate=ratedate
        # self.tempdate=tempdate
        self.n,self.B=self.cal_n_B()
        self.m=self.cal_m()
        self.ratedate=ratedate
        C=self.cal_C()
        # self.ratePara=self. rate_para()

    def cal_n_B(self):
        y = self.staticStress
        x = self.staticStrain

        def f(X,B,n):#输入准静态下应力和应变数据，返回拟合过后的模型

            return self.yieldstress+B*X**n

        popt, pcov = curve_fit(f, x, y, maxfev=500000)
        B, n=popt
        pre=f(x,B,n)
        plt.plot(x,pre)
        plt.plot(x,y)
        plt.show()
        print(B,n)
        return n,B

    def cal_m(self):
        strain, rate, temp, stress=np.array(self.tempdata)
        x=temp.reshape(-1)
        y=stress.reshape(-1)/(self.yieldstress+self.B*strain.reshape(-1)**self.n)
        def f(x,m):
            return 1-((x-293)/640)**m

        popt, pcov = curve_fit(f, x, y, maxfev=500000)
        m=popt
        pre=(self.yieldstress+self.B*strain**self.n)*f(temp,m)
        autoPlot_allInOne(strain, stress, strain, pre, [293,373,473])

        return m
    def cal_C(self):
        strain, rate, temp, stress = np.array(self.ratedate)
        x=rate.reshape(-1)
        y=stress.reshape(-1)/(self.yieldstress+self.B*strain.reshape(-1)**self.n)
        def f(x,C):
            return 1+C*np.log(x/0.001)

        popt, pcov = curve_fit(f, x, y, maxfev=500000)
        C = popt
        pre = (self.yieldstress + self.B * strain ** self.n) * f(rate, C)
        evalve=result_show.constituticveResult(stress,pre)
        autoPlot_allInOne(strain, stress, strain, pre, [0.001,0.01,0.1,1000,2000,3000])

        return C


staticStrain,staticStress=data.get20Date()
tempdata=data.date_diff_temp()
ratedata=data.date_diff_rate()

a=CS_model(staticStrain[5:],staticStress[5:],112,tempdata,ratedata)







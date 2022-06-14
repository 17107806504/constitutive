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
class Tm(object):
    def __init__(self,static,diff_date):
        self.staticStrain,self.staticStress,self.staticRate=np.array(static)
        self.diff_rate_strain,self.diff_rate_stress,self.diff_rate=diff_date
        a, m1, b, B, m2=self.cal_para()
        Tm.result([a, m1, b, B, m2], static,
                  diff_date)

    def cal_para(self):
        strain = []
        stress = []
        rate = []
        staticStress=[]
        for i in np.arange(len(self.diff_rate_strain)):
            y = np.interp(self.staticStrain, self.diff_rate_strain[i], self.diff_rate_stress[i])
            stress.append(y)
            staticStress.append(self.staticStress)

            strain.append(self.staticStrain)
            rate.append(np.ones(len(self.diff_rate_strain[i]))*self.diff_rate[i][0])
        strain=np.array(strain).reshape(-1)
        stress=np.array(stress).reshape(-1)
        rate=np.array(rate).reshape(-1)
        staticStress=np.array(staticStress).reshape(-1)

        def func(X, a, m1, b,B,m2):
            strain,staticStress,rate=X
            return staticStress+(a * strain ** m1 + b)*(1-staticStress/220)*np.log(rate/0.001)+B*(rate)**m2




        popt, pcov = curve_fit(func, (strain,staticStress,rate), stress, maxfev=500000)

        a, m1, b,B,m2 = popt
        return a, m1, b,B,m2

    @staticmethod
    def get_first_part(a, m, b, strain, rate, staticStress):
        return (np.array(strain) ** m * a + b) * (1 - np.array(staticStress) / 220) * np.log(
            rate / 0.001) + staticStress

    @staticmethod
    def result(para, static, diff_rate_data):
        a, m1, beta, B, m2 = para
        static_strain, static_stress, staticrate = static
        diff_rate_strain, diff_rate_stress, diff_rate = diff_rate_data
        strain = []
        stress = []
        for i in np.arange(len(diff_rate)):
            strain.append(static_strain)
            y = np.interp(static_strain, diff_rate_strain[i], diff_rate_stress[i])
            stress.append(y)
        stress = np.array(stress)
        strain = np.array(strain)
        diff_rate = np.array(diff_rate)
        pre = Tm.get_first_part(a, m1, beta, strain, diff_rate, static_stress) + B * np.log(diff_rate / 0.001) ** m2
        autoPlot_allInOne(strain, stress, strain, pre, [0.01, 0.1, 1000, 2000, 3000])
        evalve = result_show.constituticveResult(stress, pre)

static=[strain[0],stress[0],rate[0]]
diff_hdate=[strain[1:],stress[1:],rate[1:]]

test=Tm(static,diff_hdate)
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:09:21 2021

@author: wpl
"""

import xlrd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import linear_model
from scipy.optimize import curve_fit
from date_function import *
from scipy.optimize import curve_fit,fsolve,leastsq


def func_line(x, a, b):
    return a * x + b


def cal_n1(rate, stress):
    y = np.log(rate)
    x = np.log(stress)
    popt, pcov = curve_fit(func_line, x, y)
    n1, o = popt
    return n1


def cal_beta(rate, stress):
    y = np.log(rate)
    x = np.array(stress)
    # print(x)
    popt, pcov = curve_fit(func_line, x, y)
    beta, o = popt
    return beta


def cal_arpha(rate, stress):
    return cal_beta(rate, stress) / cal_n1(rate, stress)


def cal_n(arpha, rate, stress):
    y = np.log(rate)
    stress = np.array(stress)
    x = np.log((np.exp(arpha * stress) - np.exp(arpha * stress * -1)) / 2)
    popt, pcov = curve_fit(func_line, x, y)
    n, o = popt

    return n


'''尝试改变arpha，但是作用不大'''


# def cal_erro(a,b,x,y):
#     erro=0
#     for i in np.arange(len(x)):
#         erro=erro+(y[i]-(a*x[i]+b))**2
#     return erro

# def find_best_n_a(rate,stress):
#     initial_arpha=cal_arpha0(rate,stress)
#     arpha=initial_arpha/10
#     Erro=[]
#     N=[]
#     for i in np.arange(20):
#         n,o,x,y=cal_n(arpha,rate,stress)
#         erro=cal_erro(n,o,x,y)
#         Erro.append(erro)
#         N.append(n)
#         arpha=arpha+initial_arpha/10
#     min_erro=min(Erro)
#     i=Erro.index(min_erro)
#     print(Erro,i, min_erro)
#     return initial_arpha/10+i*initial_arpha/10,N[i]


# print(find_best_n_a(strain_rate,getStress_of_certainstrain(strain,stress,strain_rate,0.1)))


def cal_Q(alpha, n, temp, stress,rate):
    x = 1000 / np.array(temp)
    stress = np.array(stress)
    y = n * np.log((np.exp(stress * alpha) - np.exp(stress * -1 * alpha)) / 2)

    popt, pcov = curve_fit(func_line, x, y)
    slope, inter = popt
    Q = slope * 8.314
    lnA = math.log(rate) - 1 * inter
    A=np.exp(lnA)
    return Q,A





# def cal_paraOfDiffStrain(max_strain, stress, strain, rate):
#     present_strain = 0.08
#     Arpha = []
#     N = []
#     Strain = []
#     while present_strain <= max_strain:  # 计算过程中精度丢失，导致其值变小
#         used_stress = getStress_of_certainstrain(strain, stress, rate, present_strain)
#         arpha = cal_arpha(rate, used_stress)
#         if arpha > 0:  # arpha小于零会导致整个公式有问题，去掉
#             n = cal_n(arpha, rate, used_stress)
#             Arpha.append(arpha)
#             N.append(n)
#             Strain.append(present_strain)
#         present_strain = round(present_strain + 0.01, 4)
#
#     return Strain, Arpha, N

#
#
# def func_multi(x, a, b, c):
#     return a + b * x + c * x ** 2




#
# def cal_para(strain):
#     arpha = func_multi(strain, a1, b1, c1)
#     n = func_line(strain, a2, b2)
#     Q = func_line(strain, a3, b3)
#     lnA = func_line(strain, a4, b4)
#     return arpha, n, Q, lnA


# def get_cal_stress(rate):
#     strain = 0.01
#     Stress = []
#     Strain = []
#     while strain < 0.2:
#         Strain.append(strain)
#         arpha, n, Q, lnA = cal_para(strain)
#         A = math.exp(lnA)
#         Z = rate * math.exp(Q / (8.314 * 0.293))
#         stress = math.log((Z / A) ** (1 / n) + ((Z / A) ** (2 / n) + 1) ** 0.5) / arpha
#         Stress.append(stress)
#         strain = strain + 0.01
#
#     return Strain, Stress
def dataAtCertainRate(t,stress,rate,Rate):
    xExport=[]
    yExport = []
    zExport = []
    for i in np.arange(len(t)):
        if rate[i]==Rate:
            xExport.append(t[i])
            yExport.append(stress[i])
    return xExport,yExport


def dataAtCertainT(x,y,z,T):
    xExport=[]
    yExport = []
    zExport = []
    for i in np.arange(len(x)):
        if z[i]==T:
            xExport.append(x[i])
            yExport.append(y[i])
    return xExport,yExport



def ArrheniusPara(rate,stress,t,T,Rate):

    x,y=dataAtCertainT(rate,stress,t,T)

    arpha=cal_arpha(x,y)

    n=cal_n(arpha,x,y)

    x,y=dataAtCertainRate(t,stress,rate,Rate)
    Q, A=cal_Q(arpha,n,x,y,0.001)


    return arpha,n,Q,A


def A_type(rate,T,para):
    arpha,n,Q,A=para
    rate=rate
    T=T

    def f_sol(x):
        return np.log(rate+1e-5)-np.log(A)+Q*1000/(T*8.14)-n*np.log((np.exp(x * arpha) - np.exp(x * -1 * arpha)) / 2+1e-5)

    result = fsolve(f_sol, [112])
    return result


def calZ(arpha,n, A,stress):
    return A*(((np.exp(stress * arpha) - np.exp(stress * -1 * arpha)) / 2+1e-10)**n)
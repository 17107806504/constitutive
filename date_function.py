# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 15:46:58 2021

@author: wpl
"""

import xlrd
import numpy as np



'''输出的数据为  3*6*100   的矩阵   '''
def Getdate_rate():
    x1=xlrd.open_workbook(r'E:\date\学习\origin\不同应变率.xls')
    table = x1.sheets()[0]
    
    strain_rate=[0.001,1000,2000,3000,4000,5000]
    
    j=18
    date=[]
    strain=[]
    stress=[]
    rate=[]
    for i in strain_rate:
        strain.append(table.col_values(j)[3:])
        stress.append(table.col_values(j+1)[3:])
        rate.append(i*np.ones(1000))
        j+=2
    date=[rate,strain,stress]
    return date



def Getdate_temp():
    x1=xlrd.open_workbook(r'E:\date\学习\origin\不同温度.xls')
    table = x1.sheets()[0]
    
    Temp=[293,373,473,573]
    
    j=12
    date=[]
    strain=[]
    stress=[]
    temp=[]
    for i in Temp:
        strain.append(table.col_values(j)[3:])
        stress.append(table.col_values(j+1)[3:])
        temp.append(i*np.ones(1000))
        j+=2
    date=[temp,strain,stress]
  
    return date


'''通用的数据'''
def getdateCommon(path,col,rowStart,rowTo):
    x1 = xlrd.open_workbook(path)
    table = x1.sheets()[0]
    date = table.col_values(col)[rowStart:rowTo]
    return date

def calTrueValue(path,col,rowStart,rowTo):
    load =getdateCommon(path, col, rowStart, rowTo)
    displacement = getdateCommon(path, col+1, rowStart, rowTo)
    eStrain = np.array(displacement) / 50
    eStrainPlus1 = eStrain + 1
    # trueStrain = np.log(eStrainPlus1)
    eStress = np.array(load) * 1000 / 15
    # trueStress = eStrainPlus1 * eStress
    return eStrain,eStress
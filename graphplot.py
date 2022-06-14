# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:40:57 2021

@author: wpl
"""

import matplotlib.pyplot as plt
import numpy as np
from date_function import *
from matplotlib.font_manager import FontProperties
plt.rcParams["font.sans-serif"] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False

font1 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=20)
font2 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=12)
font3 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=8)

def autoPlot_test(x,y,label,title):
    fig=plt.figure()
    plt.xlim(0, 0.4)
    plt.ylim(0, 350)
    for i in np.arange(len(label)):
        l=str(label[i])
        plt.plot(x[i],y[i],label='0.001'+r'${s^{-1}}$'+', '+l+r'$^{\circ}$C')
    plt.xlabel('strain ',fontproperties=font1)
    plt.ylabel('stress (MPa)',fontproperties=font1)

    plt.xticks(fontproperties=font2)
    plt.yticks(fontproperties=font2)
    plt.title(title)
    plt.legend()
    plt.show()
        
        
        
# label,x,y=Getdate_rate()
# autoPlot_test(x,y,label)




def autoPlot_twoInOne(x,y,x_numeric,y_numeric,label):
    color=['b','g','r','c','m','y']
    for i in np.arange(len(label)):
        fig=plt.figure()
        l=str(label[i][0])
        plt.plot(x[i],y[i],label=l,color=color[i])
        plt.plot(x_numeric[i],y_numeric[i],linestyle='-.',label='num_'+l,color=color[i])
        plt.xlabel('strain')
        plt.ylabel('stress')
        plt.legend()
        
    
    
def autoPlot_allInOne(x,y,x_numeric,y_numeric,label):
    fig=plt.figure()
    plt.xlim(0,0.4)
    plt.ylim(0,350)
    color=['b','g','r','c','m','y','orange','grey']
    for i in np.arange(len(label)):
        
       
        l=str(label[i])
        plt.plot(x[i],y[i],label=l+r'${s^{-1}}$',color=color[i])
        plt.plot(x_numeric[i],y_numeric[i],linestyle='-.',label='pre_'+l+r'${s^{-1}}$',color=color[i])
    plt.xlabel('strain ',fontproperties=font1)
    plt.ylabel('stress (MPa)',fontproperties=font1)

    plt.xticks(fontproperties=font2)
    plt.yticks(fontproperties=font2)

    plt.legend()
    plt.show()
    
    
    
    
    
    
def autoPlot_test_slope(x,y,label):
    fig=plt.figure()
    
    for i in np.arange(len(label)):
        l=str(label[i][0])
        slope=[]
        x_slope=[]
        for j in np.arange(len(x[i])):
            if j and y[i][j]>100 :
                slope.append((y[i][j]-y[i][j-1])/(x[i][j]-x[i][j-1]))
                x_slope.append(x[i][j])
        plt.scatter(x_slope,slope,label=l)

    plt.xlabel('strain')
    plt.ylabel('stress')
   
    plt.legend()



   
# autoPlot_test_slope(x,y,label)   
    
    
def relevance(y,y_num,size):
    y.reshape(size)
    y_num.reshape(size)
    fig=plt.figure()
    plt.xlim(50,400)
    plt.ylim(50,400)
    plt.scatter(y, y_num)
    plt.plot(y, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import matplotlib.pyplot as plt
import numpy as np #导入numpy工具包
import xlrd
import date_function as df
import graphplot as plts2
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit,fsolve,leastsq

def dele_neg(x,y):
    x_out = []
    y_out = []
    for i in np.arange(len(y)):
        if y[i] > 0 and x[i] > 0:
            x_out.append(x[i])
            y_out.append(y[i])
    return x_out,y_out


def get20Date():
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-20-1.xls'
    trueStrain1,trueStress1=df.calTrueValue(path,1,22,4000)
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-20-2.xls'
    trueStrain2,trueStress2=df.calTrueValue(path,1,22,4000)

    #plts2.autoPlot_test([trueStrain1,trueStrain2],[trueStress1,trueStress2],['1','2'],title='20度 0.001 1/s')
    x=[]

    trueStrain1, trueStress1=dele_neg(trueStrain1, trueStress1)
    trueStrain2, trueStress2 = dele_neg(trueStrain2, trueStress2)

    for i in range(50):
        x.append(0.02+i*0.25/50)
    y1=np.interp(x, trueStrain1, trueStress1)
    y2=np.interp(x, trueStrain2, trueStress2)
    y=(np.array(y1)+np.array(y2))/2
    x=np.array(x)-0.019

    return np.array(x),np.array(y)




def get100Date():
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-100-1.xls'
    trueStrain1,trueStress1=df.calTrueValue(path,1,22,5000)
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-100-2.xls'
    trueStrain2,trueStress2=df.calTrueValue(path,1,22,5000)


    x=[]
    trueStrain1, trueStress1 = dele_neg(trueStrain1, trueStress1)
    trueStrain2, trueStress2 = dele_neg(trueStrain2, trueStress2)

    for i in range(50):
        x.append(0.02+i*0.32/50)
    y1=np.interp(x, trueStrain1, trueStress1)
    y2=np.interp(x, trueStrain2, trueStress2)
    y=(np.array(y1)+np.array(y2))/2
    x = np.array(x) - 0.019


    return np.array(x),np.array(y)





def get200Date():
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-200-1.xls'
    trueStrain1,trueStress1=df.calTrueValue(path,1,22,4000)
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-200-2.xls'
    trueStrain2,trueStress2=df.calTrueValue(path,1,22,4000)
    #plts2.autoPlot_test([trueStrain1,trueStrain2],[trueStress1,trueStress2],['1','2'],title='20度 0.001 1/s')
    trueStrain1, trueStress1 = dele_neg(trueStrain1, trueStress1)
    trueStrain2, trueStress2 = dele_neg(trueStrain2, trueStress2)

    x=[]

    for i in range(50):
        x.append(0.02+i*0.41/50)
    y1=np.interp(x, trueStrain1, trueStress1)
    y2=np.interp(x, trueStrain2, trueStress2)
    y=(np.array(y1)+np.array(y2))/2

    x=np.array(x)-0.019

    return np.array(x),np.array(y2)


def get300Date():
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-300-1.xls'
    trueStrain1,trueStress1=df.calTrueValue(path,1,22,5000)
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-300-2.xls'
    trueStrain2,trueStress2=df.calTrueValue(path,1,22,5000)

    trueStrain1, trueStress1 = dele_neg(trueStrain1, trueStress1)
    trueStrain2, trueStress2 = dele_neg(trueStrain2, trueStress2)
    x=[]

    for i in range(50):
        x.append(0.045+i*0.15/50)
    y1=np.interp(x, trueStrain1, trueStress1)
    y2=np.interp(x, trueStrain2, trueStress2)
    y=(np.array(y1)+np.array(y2))/2
    x = np.array(x) - 0.044



    return np.array(x), np.array(y)


def get1000RateDate():
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-1000.xls'
    trueStrain=df.getdateCommon(path,0,1,1709)
    trueStress=df.getdateCommon(path,1,1,1709)
    trueStrain, trueStress = dele_neg(trueStrain, trueStress)

    x=[]
    for i in range(50):
        x.append(0.0002+i*0.155/50)
    y=np.interp(x, trueStrain, trueStress)

    return np.array(x),np.array(y)


def get2000RateDate():
    path = r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-2000.xls'
    trueStrain = df.getdateCommon(path, 0, 1, 1817)
    trueStress = df.getdateCommon(path, 1, 1, 1817)
    trueStrain, trueStress = dele_neg(trueStrain, trueStress)

    x = []
    for i in range(50):
        x.append(0.0002+i * 0.127 / 50)
    y = np.interp(x, trueStrain, trueStress)


    return np.array(x),np.array(y)


def get2000RateDate_fit():
    path = r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-2000.xls'
    trueStrain = df.getdateCommon(path, 0, 1, 1817)
    trueStress = df.getdateCommon(path, 1, 1, 1817)
    trueStrain, trueStress = dele_neg(trueStrain, trueStress)

    x = []
    for i in range(50):
        x.append(0.0002+i * 0.127 / 50)
    y = np.interp(x, trueStrain, trueStress)

    def f(x, a, b, c):
        return (a + b * x + c * x ** 2)

    p0 = [180, 1, 1]
    popt, pcov = curve_fit(f, x, y, p0, maxfev=5000000)
    coe1, coe2, c = popt
    print(popt)
    y = coe1 + coe2 * np.array(x) + c * np.array(x) ** 2


    return np.array(x), np.array(y)


def get3000RateDate():
    path = r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-3000.xls'
    trueStrain= df.getdateCommon(path, 0, 1, 1814)
    trueStress= df.getdateCommon(path, 1, 1, 1814)
    trueStrain, trueStress = dele_neg(trueStrain, trueStress)

    x = []
    for i in range(50):
        x.append(0.0002+i * 0.2 / 50)
    y = np.interp(x, trueStrain, trueStress)


    return np.array(x ),np.array(y)


def get4000RateDate():
    path = r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-4000.xls'
    trueStrain = df.getdateCommon(path, 0, 1, 1771)
    trueStress = df.getdateCommon(path, 1, 1, 1771)
    trueStrain, trueStress = dele_neg(trueStrain, trueStress)

    x = []
    for i in range(50):
        x.append(i * 0.235 / 50)
    y = np.interp(x, trueStrain, trueStress)
    #plts2.autoPlot_test([x], [y], ['ye'], title='4000 1/s')
    def f(x,a,b,c):

        return (a+b*x+c*x**2)
    p0=[180,1,1]
    popt, pcov=curve_fit(f,x,y,p0,maxfev=5000000)
    coe1,coe2,c= popt
    y=coe1+coe2*np.array(x)+c*np.array(x)**2


    return np.array(x),np.array(y)


def get0_01RateDate():
    path = r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-0.01.xls'
    trueStrain = df.getdateCommon(path, 0, 1, 10)
    trueStress = df.getdateCommon(path, 1, 1, 10)

    x = []
    for i in range(50):
        x.append(0.001+i * (trueStrain[len(trueStrain)-1]-trueStrain[0])/ 50)
    y = np.interp(x, trueStrain, trueStress)

    x_out = []
    y_out = []
    for i in np.arange(len(y)):
        if y[i] > 0 and x[i]>0:
            x_out.append(x[i])
            y_out.append(y[i])

    return np.array(x_out), np.array(y_out)


def get0_1RateDate():
    path = r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052-20-0.1.xls'
    trueStrain = df.getdateCommon(path, 0, 1, 9)
    trueStress = df.getdateCommon(path, 1, 1, 9)

    x = []
    for i in range(50):
        x.append(0.001+i * (trueStrain[len(trueStrain)-1]-trueStrain[0])/ 50)
    y = np.interp(x, trueStrain, trueStress)

    x_out = []
    y_out = []
    for i in np.arange(len(y)):
        if y[i] > 0 and x[i]>0:
            x_out.append(x[i])
            y_out.append(y[i])

    return np.array(x_out), np.array(y_out)


def date_diff_rate():
    strain = [list(get20Date()[0]) , list(get0_01RateDate()[0]) , list(get0_1RateDate()[0]) ,list(
        get1000RateDate()[0]) , list(get2000RateDate()[0]) , list(get3000RateDate()[0])]
    stress = [list(get20Date()[1]) , list(get0_01RateDate()[1]) , list(get0_1RateDate()[1]) , list(
        get1000RateDate()[1]) , list(get2000RateDate()[1]) , list(get3000RateDate()[1])]
    rate = [list(np.ones(len(get20Date()[0])) * 0.001) , list(np.ones(len(get0_01RateDate()[0])) * 0.01) , list(
        np.ones(len(get0_1RateDate()[0])) * 0.1) , list(np.ones(len(get1000RateDate()[0])) * 1000) , list(
        np.ones(len(get2000RateDate()[0])) * 2000)
            ,list(np.ones(len(get3000RateDate()[0])) * 3000)]
    temp = np.array(list(np.ones(len(stress[1])) * 293)*len(stress)).reshape(6,-1)
    return strain,rate,temp,stress



def date_diff_condition():
    strain = [get20Date()[0],get0_01RateDate()[0],get0_1RateDate()[0], get1000RateDate()[0] ,get2000RateDate()[0], get3000RateDate()[0]]
    stress = [get20Date()[1],get0_01RateDate()[1],get0_1RateDate()[1], get1000RateDate()[1], get2000RateDate()[1] ,get3000RateDate()[1]]
    temp=[list(np.ones(len(get20Date()[0]))*293),list(np.ones(len(get20Date()[0]))*293),list(np.ones(len(get20Date()[0]))*293),list(np.ones(len(get20Date()[0]))*293)
        ,list(np.ones(len(get20Date()[0]))*293),list(np.ones(len(get20Date()[0]))*293),list(np.ones(len(get100Date()[0]))*373),list(np.ones(len(get200Date()[0]))*473)]
    strain.append(get100Date()[0])
    strain.append(get200Date()[0])
    stress.append(get100Date()[1])
    stress.append(get200Date()[1])
    rate=[list(np.ones(len(get20Date()[0]))*0.001),list(np.ones(len(get0_01RateDate()[0]))*0.01),list(np.ones(len(get0_1RateDate()[0]))*0.1),list(np.ones(len(get1000RateDate()[0]))*1000),list(np.ones(len(get2000RateDate()[0]))*2000)
          ,list(np.ones(len(get3000RateDate()[0]))*3000),list(np.ones(len(get100Date()[0]))*0.001),list(np.ones(len(get200Date()[0]))*0.001)]

    return strain,rate,temp, stress

def date_diff_temp():
    strain=[get20Date()[0],get100Date()[0],get200Date()[0]]
    stress= [get20Date()[1],get100Date()[1],get200Date()[1]]
    temp=[np.ones(len(get20Date()[0]))*293,np.ones(len(get20Date()[0]))*373,np.ones(len(get20Date()[0]))*473]
    rate=np.ones(shape=np.array(stress).shape)*0.001
    return strain, rate, temp, stress


if __name__=='__main__':
    # x,y=get4000RateDate()
    # plts2.autoPlot_test([x], [y], ['mean'], title='300度 0.001 1/s')
    x,y2=get2000RateDate_fit()
    x,y=get2000RateDate()
    plt.plot(x,y)
    plt.plot(x,y2)
    plt.show()

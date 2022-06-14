import numpy as np #导入numpy工具包
import xlrd
import date_function as df
import graphplot as plts2



def get100Date():
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-100-1.xls'
    trueStrain1,trueStress1=df.calTrueValue(path,1,22,5000)
    path=r'E:\date\科研\数据收集\5052\5052等温拉伸-准静态应变率0.001\5052等温拉伸-准静态应变率0.001\5052-100-2.xls'
    trueStrain2,trueStress2=df.calTrueValue(path,1,22,5000)

    #plts2.autoPlot_test([trueStrain1,trueStrain2],[trueStress1,trueStress2],['1','2'],title='100度 0.001 1/s')
    x=[]

    for i in range(100):
        x.append(0.02+i*0.0032)
    y1=np.interp(x, trueStrain1, trueStress1)
    y2=np.interp(x, trueStrain2, trueStress2)
    y=(np.array(y1)+np.array(y2))/2

    #plts2.autoPlot_test([x],[y],['mean'],title='100度 0.001 1/s')
    return x,y
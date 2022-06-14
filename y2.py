import xlrd
import numpy as np
from date_function import *
from data import *
import Arrhenius

from scipy.optimize import curve_fit,fsolve,leastsq
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

strain20,stress20=get20Date()
strain100,stress100=get100Date()
strain200,stress200=get200Date()
strain300,stress300=get300Date()

stress1=[stress20[0],stress100[0],stress200[0],stress300[0],121.8,122]
temp1=[293,373,473,573,293,293]
strainRate1=[0.001,0.001,0.001,0.001,0.01,0.1]
highRate=[1000,2000,3000]
highRateStress=[get1000RateDate()[1][0],get2000RateDate_fit()[1][0],get3000RateDate()[1][0]]






def yieldmodel():


    strainRate2=strainRate1+highRate
    stress2=stress1+highRateStress
    temp2=temp1+3*[293]

    arpha,n,Q, A=Arrhenius.ArrheniusPara(strainRate1,stress1,temp1,293,0.001)




    def getN():
        N=[]
        for i in np.arange(len(highRate)):
            stress=highRateStress[i]
            rate=highRate[i]

            def theN():

                BdividedbyT = 5.36e-14
                b = 2.86e-10
                m = 3.06
                T = 293


                return rate*BdividedbyT*T/(b**2*(stress/m))

           # print(rate*BdividedbyT*293/(b*(stress-m*30)))
            result = theN()
            N.append(result)
        return N

    def f(x,a,b,c):

        return (a+b*np.log10(x/0.001)**c)





    rate=[1000, 2000, 3000]
    x=np.array(rate)
    y=getN()
    y=np.array(y)*2.86e-10
    popt, pcov=curve_fit(f,x,y,maxfev=5000000)
    coe1,coe2,coe3= popt



    tradition = [arpha, n, Q, A]
    densCoe=coe1,coe2,coe3


    return tradition,densCoe

[arpha, n, Q, A],[coe1,coe2,coe3]=yieldmodel()






def solveStress(rate,T):
    dens = (coe1 + coe2 * np.log10(rate / 0.001) ** coe3)
    BdividedbyT = 5.36e-14
    B = BdividedbyT * T
    b = 2.86e-10
    m = 3.06
    T = T
    initial = 80

    x = 0
    while x < rate:
        x = dens / (B / (b * (initial / m )) + dens / (
                    Arrhenius.calZ(arpha, n, A, initial) / np.exp(Q * 1000 / (8.14 * T))))
        initial += 0.001
        if initial > 300:
            break

    return initial

if __name__=='__main__':
    strainpre = [0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 500, 800] + [1000,1100,1200,1500,1800,2000,2200,2500,2800,3000]
    strainRateX = [0.001, 0.01, 0.1] + highRate
    stressy = [stress20[0], 121.8, 122] + [get1000RateDate()[1][0],get2000RateDate()[1][0],get3000RateDate()[1][0]]

    stressPre = []

    for i in np.arange(len(strainpre)):
        stressPre.append(solveStress(strainpre[i],293))


    stressPre3 = []
    density = []
    for i in strainpre:
        stressPre3.append(Arrhenius.A_type(i, 293, [arpha, n, Q, A]))
        density.append((coe1 + coe2 * np.log10(i / 0.001) ** coe3) / 2.86e-10)

    font1 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=20)
    font2 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=12)
    font3 = FontProperties(fname=r"C:\Windows\Fonts\times.TTF", size=8)
    x = np.log10(strainRateX)
    x1 = np.log10(strainpre)
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(x1, stressPre, label='Arrhenius扩展模型预测值')
    # plt.plot(x1,stressPre2,label='Arrhenius模型全数据预测值')
    plt.plot(x1, stressPre3, label='Arrhenius模型预测值')
    plt.scatter(x, stressy, color='r', label='真实值', )
    plt.ylim(0)
    plt.xlabel(r"log($\dot{\varepsilon}$)",fontproperties=font1)
    plt.ylabel('stress (Mpa)',fontproperties=font1)
    plt.xticks(fontproperties=font2)
    plt.yticks(fontproperties=font2)
    plt.title('流动应力预测')
    plt.legend()
    plt.show()

    plt.plot(x1, density, label='可动位错密度预测值')
    plt.xlabel(r"log($\dot{\varepsilon}$)",fontproperties=font1)
    plt.ylabel('mobile dislocation density'+r'${(m^{-2})}$',fontproperties=font1)
    plt.title('可动位错密度预测')
    plt.legend()
    plt.show()







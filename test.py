import matplotlib.pyplot as plt

import y2
import data
import graphplot as plts2
import numpy as np
from scipy.optimize import curve_fit
import graphplot
import result_show



def selectPositive(x,y,rate,T):
    y=y-y2.solveStress(rate,T)
    index=[]
    for i in np.arange(len(x)):
        if y[i]<0:
            index.append(i)
    y = np.delete(y, index)
    x = np.delete(x, index)
    # plt.plot(x,y)
    # plt.show()
    return x,y

def getArgument(x,y):
    dStrain=[]
    dDensity=[]
    Density=(y/(0.5*3.06*26654*2.86e-10))**2
    strain1=x[0]
    density1=Density[0]
    Density_o=[]
    for i in np.arange(len(x)-1):
        if (Density[i+1]-density1)>0:
            dStrain.append(x[i+1]-strain1)
            strain1=x[i+1]
            dDensity.append(Density[i+1]-density1)
            Density_o.append(density1)
            density1=Density[i+1]
    Density= np.delete(Density,len(Density)-1)
    densDiff=np.array(dDensity)/np.array(dStrain)

    return densDiff,Density_o




def dislocationEvo(X,k1,k2):
    Density=X
    return k1*Density**0.5-k2*Density

def getTwoK(strain,stress,rate,T):
    x, y = selectPositive(strain, stress, rate,T)
    densDiff, Density = getArgument(x, y)
    p0=[214896524.04856375 ,27.122720955547734]
    popt, pcov = curve_fit(dislocationEvo, Density, densDiff,bounds=([0,0],[np.inf,np.inf]))

    return popt
strain20,stress20=data.get20Date()





strain100,stress100=data.get100Date()
strain200,stress200=data.get200Date()
strain300,stress300=data.get300Date()
strain1000,stress1000=data.get1000RateDate()
strain2000,stress2000=data.get2000RateDate()
strain2000,stress2000_f=data.get2000RateDate_fit()
strain3000,stress3000=data.get3000RateDate()

strain0_01,stress0_01=data.get0_01RateDate()
strain0_1,stress0_1=data.get0_1RateDate()


t=[293,373,473,293,293,293,293,293]
rate=[0.001,0.001,0.001,0.01,0.1,1000,2000,3000]
k1=[getTwoK(strain20,stress20,0.001,293)[0],getTwoK(strain100,stress100,0.001,373)[0],getTwoK(strain200,stress200,0.001,473)[0],
    getTwoK(strain0_01,stress0_01,0.01,293)[0],getTwoK(strain0_1,stress0_1,0.1,293)[0],getTwoK(strain1000,stress1000,1000,293)[0],getTwoK(strain2000,stress2000_f,2000,293)[0],
    getTwoK(strain3000,stress3000,3000,293)[0]]
k2=[getTwoK(strain20,stress20,0.001,293)[1],getTwoK(strain100,stress100,0.001,373)[1],getTwoK(strain200,stress200,0.001,473)[1],
    getTwoK(strain0_01,stress0_01,0.01,293)[1],getTwoK(strain0_1,stress0_1,0.1,293)[1],getTwoK(strain1000,stress1000,1000,293)[1],getTwoK(strain2000,stress2000_f,2000,293)[1],
    getTwoK(strain3000,stress3000,3000,293)[1]]

def fit_k(X,A,n,Q):
    rate,T=X
    rate=np.array(rate)
    T=np.array(T)
    return A*(1e-6+np.log10(rate/0.001)*np.exp(Q*1000/(8.14*T)))**n
X=[rate,t]
popt, pcov = curve_fit(fit_k,X, k1,bounds=([-np.inf,-0.01,0],[np.inf,0.1,1000]),maxfev=500000)
A1,n1,Q1=popt
popt, pcov = curve_fit(fit_k,X, k2,bounds=([-np.inf,-0.01,1000],[np.inf,0.1,np.inf]),maxfev=500000)
A2,n2,Q2=popt

print([A1,n1,Q1],[A2,n2,Q2])

def cal_stress_at_strain(strain,rate,T):
    k1 = fit_k([rate, T], A1, n1, Q1)
    k2 = fit_k([rate, T], A2, n2, Q2)



    # k1=getTwoK(strain20, stress20, 0.001, 293)[0]
    # k2 = getTwoK(strain20, stress20, 0.001, 293)[1]

    Density = []
    strain1 = 0
    density1 = 1e10
    stress_at_strain=[]
    for i in np.arange(len(strain)):
        dstrain=strain[i]-strain1
        strain1=strain[i]
        densDiff=dislocationEvo(density1,k1,k2)
        density1=density1+densDiff*dstrain
        Density.append(density1)
        stress_at_strain.append(0.5*3.06*26654*2.86e-10*density1**0.5)
        #print(0.5*3.06*26654*2.86e-10*density1**0.5)

    return stress_at_strain,Density

# stress_at_strain,Density=cal_stress_at_strain(strain20,0.001,293)
# stress_at_strain=np.array(stress_at_strain)+yieldstress.solveStress(0.001,293)
# plt.plot(strain20,stress_at_strain)
# plt.plot(strain20,stress20)
# plt.show()

strain_diff_rate=[strain20,strain0_01,strain0_1,strain1000,strain2000,strain3000]
stress_diff_rate=[stress20,stress0_01,stress0_1,stress1000,stress2000,stress3000]
diff_rate=[0.001,0.01,0.1,1000,2000,3000]
stress_diff_rate_pre=[]
for i in np.arange(len(strain_diff_rate)):
    stress_at_strain, Density = cal_stress_at_strain(strain_diff_rate[i], diff_rate[i], 293)
    stress_at_strain = np.array(stress_at_strain) + y2.solveStress(diff_rate[i], 293)
    stress_diff_rate_pre.append(stress_at_strain)
#
#graphplot.autoPlot_test(strain_diff_rate,stress_diff_rate,[0.001,0.01,0.1,1000,2000,3000,4000],'不同应变率下的实验数据')

graphplot.autoPlot_allInOne(strain_diff_rate,stress_diff_rate,strain_diff_rate,stress_diff_rate_pre,diff_rate)

evalve=result_show.constituticveResult(np.array(stress_diff_rate),np.array(stress_diff_rate_pre))
strain_diff_tem=[strain20,strain100,strain200]
stress_diff_tem=[stress20,stress100,stress200]
diff_tem=[293,373,473]
stress_diff_tem_pre=[]
for i in np.arange(len(strain_diff_tem)):
    stress_at_strain, Density = cal_stress_at_strain(strain_diff_tem[i], 0.001, diff_tem[i])
    stress_at_strain = np.array(stress_at_strain) + y2.solveStress(0.001, diff_tem[i])
    stress_diff_tem_pre.append(stress_at_strain)
#graphplot.autoPlot_test(strain_diff_tem,stress_diff_tem,[20,100,200],'不同温度下的实验数据')

graphplot.autoPlot_allInOne(strain_diff_tem,stress_diff_tem,strain_diff_tem,stress_diff_tem_pre,diff_tem)

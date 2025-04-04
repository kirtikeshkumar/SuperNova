import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d, UnivariateSpline,InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from scipy.optimize import minimize
from matplotlib.ticker import LogLocator

path = "data/Livermore/"
nu_e_L = np.loadtxt(path+"Nu_e_Luminosity.dat")
anu_e_L = np.loadtxt(path+"aNu_e_Luminosity.dat")
nu_x_L = np.loadtxt(path+"Nu_x_Luminosity.dat")
nu_e_avE = np.loadtxt(path+"Nu_e_avE.dat")
anu_e_avE = np.loadtxt(path+"aNu_e_avE.dat")
nu_x_avE = np.loadtxt(path+"Nu_x_avE.dat")
fnue_L = InterpolatedUnivariateSpline(nu_e_L[:,0], nu_e_L[:,1], k = 1)
fanue_L = InterpolatedUnivariateSpline(anu_e_L[:,0], anu_e_L[:,1], k = 1)
fnux_L = InterpolatedUnivariateSpline(nu_x_L[:,0], nu_x_L[:,1], k = 1)
fnue_E = InterpolatedUnivariateSpline(nu_e_avE[:,0], nu_e_avE[:,1], k = 1)
fanue_E = InterpolatedUnivariateSpline(anu_e_avE[:,0], anu_e_avE[:,1], k = 1)
fnux_E = InterpolatedUnivariateSpline(nu_x_avE[:,0], nu_x_avE[:,1], k = 1)


T_nu = np.append(np.linspace(0.02,0.1,41),np.linspace(0.11,16,1590))
nue_L_net=quad(fnue_L,T_nu[0],T_nu[-1],epsrel=1e-4)[0]*1e50
anue_L_net=quad(fanue_L,T_nu[0],T_nu[-1],epsrel=1e-4)[0]*1e50
nux_L_net=quad(fnux_L,T_nu[0],T_nu[-1],epsrel=1e-4)[0]*1e50

nue_E_net=quad(fnue_E,T_nu[0],T_nu[-1],epsrel=1e-4)[0]/(T_nu[-1]-T_nu[0])
anue_E_net=quad(fanue_E,T_nu[0],T_nu[-1],epsrel=1e-4)[0]/(T_nu[-1]-T_nu[0])
nux_E_net=quad(fnux_E,T_nu[0],T_nu[-1],epsrel=1e-4)[0]/(T_nu[-1]-T_nu[0])

print("Luminosity nu_e: ", nue_L_net)
print("Luminosity anu_e: ", anue_L_net)
print("Luminosity nu_x: ", nux_L_net)

print("AvE nu_e: ", nue_E_net)
print("AvE anu_e: ", anue_E_net)
print("AvE nu_x: ", nux_E_net)

def diffflux(E,av_nu_E,TFac,TTilFac,eta,Scale=1,d=196,rs=10):
    '''
    E: Energy of neutrino
    av_nu_E: average Energy of neutrino over emission
    TFac: Temperature factor to convert from energy to temperature
    Scale: to normalise the flux
    d: distance to supernova in parsec
    rs: isothermal sphere radius
    '''
    prefac = rs**2/(6*np.sqrt(3)*np.pi**2)/d**2
    T = av_nu_E/TFac
    T_til = T*TTilFac
    exppart=1.0+np.exp(eta-E/(2*T_til))
    res = prefac*E*T_til*np.log(exppart)
    return res

def diffbb(E,av_nu_E,d=196,rs=10):
    T = av_nu_E/3.15
    prefac = 1.0#rs**2/(8*np.pi**2*d**2)
    epart = E**2/(np.exp(E/T)+1)
    return prefac*epart

bb = lambda E: diffbb(E,nue_E_net)
E_nu = np.linspace(0.01,50)
bb_net=quad(bb,E_nu[0],E_nu[-1],epsrel=1e-4)[0]
print("bb: ", bb_net*45.96e41)
##diffflux=np.vectorize(diffflux)
##diffbb = np.vectorize(diffbb)
##
##E_nu = np.linspace(0.01,30)
##fl = diffflux(E_nu,nue_E_net,3.48,0.828,0)
##bb = diffbb(E_nu,nue_E_net)
##plt.plot(E_nu,fl)
##plt.plot(E_nu,bb)
##plt.show()


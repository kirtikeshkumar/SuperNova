import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d, UnivariateSpline,InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from scipy.optimize import minimize

path = "data/Livermore/"
nu_e_L = np.loadtxt(path+"Nu_e_Luminosity.dat")
anu_e_L = np.loadtxt(path+"aNu_e_Luminosity.dat")
nu_x_L = np.loadtxt(path+"Nu_x_Luminosity.dat")
nu_e_avE = np.loadtxt(path+"Nu_e_avE.dat")
anu_e_avE = np.loadtxt(path+"aNu_e_avE.dat")
nu_x_avE = np.loadtxt(path+"Nu_x_avE.dat")

fnue_L = InterpolatedUnivariateSpline(nu_e_L[:,0], nu_e_L[:,1]*1e50, k = 1)
fanue_L = InterpolatedUnivariateSpline(anu_e_L[:,0], anu_e_L[:,1]*1e50, k = 1)
fnux_L = InterpolatedUnivariateSpline(nu_x_L[:,0], nu_x_L[:,1]*1e50, k = 1)

fnue_E = InterpolatedUnivariateSpline(nu_e_avE[:,0], nu_e_avE[:,1], k = 1)
fanue_E = InterpolatedUnivariateSpline(anu_e_avE[:,0], anu_e_avE[:,1], k = 1)
fnux_E = InterpolatedUnivariateSpline(nu_x_avE[:,0], nu_x_avE[:,1], k = 1)


def diffflux(E,t,f_E0,f_L):
    """
    Function to calculate the neutrino flux at
    E: Energy of Nuetrino
    t: time
    f_E0: function giving average Energy at time t
    f_L: function giving Luminosity at time t
    """    
    T = f_E0(t)/3.1514                       ## in MeV
    F3 = 5.6822                              ## integral _0^inf (x^3)/(exp(x)-1)dx
    bet = 1.0/T
    erg2MeV = 624151
    L = erg2MeV * f_L(t)                     ## convert from erg/s to MeV/s
    prefac = L/((T**4.0)*F3)
    FD = (E**2.0)/(np.exp(bet*E)+1)          ## Fermi-Dirac part
    return prefac*FD

diffFlux = np.vectorize(diffflux)

E_nu = np.linspace(0.2,50,250)
T_nu = np.append(np.linspace(0.02,0.1,41),np.linspace(0.11,16,1590))


plt.plot(T_nu, fnue_L(T_nu))
plt.plot(T_nu, fanue_L(T_nu))
plt.plot(T_nu, fnux_L(T_nu))
plt.show()

plt.semilogy(T_nu, fnue_L(T_nu))
plt.semilogy(T_nu, fanue_L(T_nu))
plt.semilogy(T_nu, fnux_L(T_nu))
plt.show()

plt.loglog(T_nu, fnue_L(T_nu))
plt.loglog(T_nu, fanue_L(T_nu))
plt.loglog(T_nu, fnux_L(T_nu))
plt.show()

########################################################################
##                   Evaluating Time Integrated Flux                  ##
########################################################################
intFlux_nue = []
intFlux_anue = []
intFlux_nux = []

for E in E_nu:
    f_intFlux_nue = lambda t:diffflux(E,t,fnue_E,fnue_L)
    f_intFlux_anue = lambda t:diffflux(E,t,fanue_E,fanue_L)
    f_intFlux_nux = lambda t:diffflux(E,t,fnux_E,fnux_L)

    intFlux_nue.append(quad(f_intFlux_nue,T_nu[0],T_nu[-1],epsrel=1e-4)[0])
    intFlux_anue.append(quad(f_intFlux_anue,T_nu[0],T_nu[-1],epsrel=1e-4)[0])
    intFlux_nux.append(quad(f_intFlux_nux,T_nu[0],T_nu[-1],epsrel=1e-4)[0])

normalisation = 1.0
data = np.loadtxt("data/supernova-spectrum_M_20_Z_2_rev_100.txt") #MeV vs No./MeV
bins=np.append(data[:,0],data[-1,1])
neutrino_flux_list = InterpolatedUnivariateSpline(np.sqrt(data[:,0]*data[:,1]), (data[:,2]+data[:,3]+data[:,4]*4)*normalisation, k = 1) #MeV vs 1/MeV/m^2
pNe = plt.hist(bins[:-1], bins, weights=data[:,2],histtype='step',ls='--',label=r'$\nu_e$',color='blue',linewidth=2)
pNbe = plt.hist(bins[:-1], bins, weights=data[:,3],histtype='step',ls='--',label=r'$\bar{\nu}_e$',color='green',linewidth=2)
pNx = plt.hist(bins[:-1], bins, weights=data[:,4],histtype='step',ls='--',label=r'$\nu_x$',color='red',linewidth=2)
pLe, = plt.plot(E_nu,intFlux_nue,label=r'$\nu_e$',color='blue',linewidth=2)
pLbe, = plt.plot(E_nu,intFlux_anue,label=r'$\bar{\nu}_e$',color='green',linewidth=2)
pLx, = plt.plot(E_nu,intFlux_nux,label=r'$\nu_x$',color='red',linewidth=2)
legcol = [pNe, pNbe, pNx,
          pLe, pLbe, pLx]
plt.xlim(0,50)
plt.xlabel("neutrino energy [MeV]", fontsize=14)
plt.ylabel("total no. of neutrino [MeV$^{-1}$]", fontsize=14)
##leg = plt.legend( fontsize=14)
##leg = plt.legend(legcol[0:6], ['', '', '', '', '', ''], ncol=2, numpoints=1, 
## borderaxespad=0., title=['Nakazato Livermore'], framealpha=.75,
## facecolor='w', edgecolor='k', loc=2, fancybox=None)
plt.show()

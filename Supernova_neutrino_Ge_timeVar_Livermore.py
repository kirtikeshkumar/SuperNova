# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
import math
from scipy.integrate import quad
from scipy.interpolate import interp1d, UnivariateSpline,InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from scipy.optimize import minimize
import os

##########################################################################
##                       Getting the data files                         ##
##########################################################################
path = "data/Livermore/"
## Luminosity in x 10^50 erg/s
nu_e_L = np.loadtxt(path+"Nu_e_Luminosity.dat")
anu_e_L = np.loadtxt(path+"aNu_e_Luminosity.dat")
nu_x_L = np.loadtxt(path+"Nu_x_Luminosity.dat")
## Average Energy in MeV
nu_e_avE = np.loadtxt(path+"Nu_e_avE.dat")
anu_e_avE = np.loadtxt(path+"aNu_e_avE.dat")
nu_x_avE = np.loadtxt(path+"Nu_x_avE.dat")
##########################################################################
##                       Define function for flux                       ##
##########################################################################
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
    res = prefac*FD
    if(res >= 0):
        return res
    else:
        return 0.0
diffFlux = np.vectorize(diffflux)
##########################################################################
##                 Define Variables and interpolations                  ##
##########################################################################
## Luminosity interpolated in erg/s
fnue_L = InterpolatedUnivariateSpline(nu_e_L[:,0], nu_e_L[:,1]*1e50, k = 1)
fanue_L = InterpolatedUnivariateSpline(anu_e_L[:,0], anu_e_L[:,1]*1e50, k = 1)
fnux_L = InterpolatedUnivariateSpline(nu_x_L[:,0], nu_x_L[:,1]*1e50, k = 1)

fnue_E = InterpolatedUnivariateSpline(nu_e_avE[:,0], nu_e_avE[:,1], k = 1)
fanue_E = InterpolatedUnivariateSpline(anu_e_avE[:,0], anu_e_avE[:,1], k = 1)
fnux_E = InterpolatedUnivariateSpline(nu_x_avE[:,0], nu_x_avE[:,1], k = 1)

E_nu = np.linspace(0.2,50,250)          ## Neutrino Energy 
T_nu = np.append(np.linspace(0.02,0.1,41),np.linspace(0.11,16,1590))        ## Time discretization
deltime_values = [0.02]+[0.5*(T_nu[i+1]-T_nu[i-1]) for i in range(1,len(T_nu)-1)]+[T_nu[-1]-T_nu[-2]]
##########################################################################

ll = 0                                  ## lower limit of time index
hl = 25                          ## upper limit of time index

pl.rcParams['font.size']=18
#from tqdm import tqdm

#Change default font size so you don't need a magnifying glass
matplotlib.rc('font', **{'size'   : 16})


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

#import CEvNS
#help(CEvNS.xsec_CEvNS)

#----Constants----
G_FERMI = 1.1664e-5     #Fermi Constant in GeV^-2
SIN2THETAW = 0.2387     #Sine-squared of the weak angle
ALPHA_EM = 0.007297353  #EM Fine structure constant
m_e = 0.5109989461e-3   #Electron mass in GeV  
SQRT2 = np.sqrt(2.0)
ee = 1.60217663E-19     #Electron charge

#----Module-scope variables----
neutrino_flux_tot = None
neutrino_flux_list = None
Enu_min = 0
Enu_max = 0
mass = 100.0 #target mass in kg

avgNumEvtGe10eVee = {}
avgNumEvtGe100eVee = {}
avgNumEvtGe170eVee = {}

avgNumEvtGe10eVth = {}
avgNumEvtGe100eVth = {}
avgNumEvtGe170eVth = {}

stdevNumEvtGe10eVee = {}
stdevNumEvtGe100eVee = {}
stdevNumEvtGe170eVee = {}

stdevNumEvtGe10eVth = {}
stdevNumEvtGe100eVth = {}
stdevNumEvtGe170eVth = {}

avgNumEvtSap50eVth = {}
avgNumEvtSap100eVth = {}

stdevNumEvtSap50eVth = {}
stdevNumEvtSap100eVth = {}



neutrino_flux_list = [lambda x: 1e-30 for i in range(6)]
##data = np.loadtxt("data/supernova-spectrum_brightest.txt") #MeV vs No./MeV
SNmass = "30"
revTime = "300"

def HelmFormFactor(E, A):
    #Define conversion factor from amu-->keV
    amu = 931.5*1e3 #keV

    #Convert recoil energy to momentum transfer q in keV
    q1 = np.sqrt(2*A*amu*E) #sqrt(keV^2) = keV

    #Convert q into fm^-1
    q2 = q1*(1e-12/1.97e-7) # fm^-1
    
    #Calculate nuclear parameters (see https://arxiv.org/abs/hep-ph/0608035)
    s = 0.9 #fm 
    a = 0.52 #fm 
    c = 1.23*(A**(1.0/3.0)) - 0.60 #fm 
    R1 = np.sqrt(c*c + 7*np.pi*np.pi*a*a/3.0 - 5*s*s) #fm
 
    #Calculate form factor
    x = q2*R1 #dimensionless
    J1 = np.sin(x)/(x*x) - np.cos(x)/x
    F = 3*J1/x
    Fq=(F*F)*(np.exp(-(q2*q2*s*s)))
##    #########################################################
##    #CONNIE
##    #########################################################
##    a=0.7e-13
##    r0=1.3e-13
##    q3=q2*1e13
##    rho0=3.0/(4.0*np.pi*r0**3)
##    R=r0*A**(1.0/3.0)
##    x1=q3*R
##    fq=(4.0*np.pi*rho0/(A*q3**3))*(np.sin(x1)-x1*np.cos(x1))*(1.0/(1.0+(a*q3)**2))
##    #print(Fq,fq,(Fq-fq)/Fq)
    return Fq

#Maximum nuclear recoil energy (in keV)
def ERmax(E_nu, A):
    #Nuclear mass in MeV
    m_A_MeV = A*0.9315e3
    #return 1e3*2.0*E_nu**2/m_N_MeV
    return 1e3*(2.0*E_nu*E_nu)/(m_A_MeV + 2*E_nu)

def xsec_CEvNS(E_R, E_nu, A, Z, gsq=0.0, m_med=1000.0):
    m_A = A*0.9315 #Mass of target nucleus (in GeV)
    q = np.sqrt(2.0*E_R*m_A) #Recoil momentum (in MeV)
    #Note: m_A in GeV, E_R in keV, E_nu in MeV
    
    #Calculate SM contribution
    Qv = (A-Z) - (1.0-4.0*SIN2THETAW)*Z #Coherence factor
    xsec_SM = (G_FERMI*G_FERMI/(4.0*np.pi))*Qv*Qv*m_A*   \
        (1.0-(q*q)/(4.0*E_nu*E_nu))
    
    #Calculate New-Physics correction from Z' coupling
    #Assume universal coupling to quarks (u and d)
    QvNP = 3.0*A*gsq

    #Factor of 1e6 from (GeV/MeV)^2
    G_V = 1 - 1e6*(SQRT2/G_FERMI)*(QvNP/Qv)*1.0/(q*q + m_med*m_med)
    
    #Convert from (GeV^-3) to (cm^2/keV)
    #and multiply by form factor and New Physics correction
    return G_V*G_V*xsec_SM*1e-6*(1.973e-14)*(1.973e-14)*HelmFormFactor(E_R, A) #1e-6 for 1/GeV to 1/keV, 1.973e-14 for 1/GeV to cm


def differentialRate_CEvNS(E_R, A, Z, gsq=0.0, m_med=1000.0):
    integrand = lambda E_nu: xsec_CEvNS(E_R, E_nu, A, Z, gsq, m_med)\
                        *neutrino_flux_Livermore(E_nu)
    E_min = np.sqrt(A*0.9315*E_R/2.0)#(E_R*1e-3+np.sqrt(2.0*A*0.9315*E_R+E_R*E_R*1e-6))/2.0
    E_min = np.maximum(E_min, Enu_min)
    #For reactor neutrinos, set E_max:
    E_max = Enu_max
    if (E_min > E_max):
        return 0
    m_N = A*1.66054e-27 #Nucleus mass in kg
    rate = quad(integrand, E_min, E_max, epsrel=1e-4)[0]/m_N
    return rate 

def differentialRate_full(E_R, A, Z, gsq=0.0, m_med=1000.0, mu_nu=0.0):
    return differentialRate_CEvNS(E_R, A, Z, gsq, m_med)



###COHERENT EVENT RATE
###Nuclear properties for Si
A_Si = 28.0
Z_Si = 14.0
##
###Nuclear properties for Ge
A_Ge = 73.0
Z_Ge = 32.0
##
###Nuclear properties for Ge
##A_Ar = 40.0
##Z_Ar = 18.0
##
##
###Nuclear properties for CsI
##A_Cs = 133.0
##Z_Cs = 55.0
A_I = 127.0
Z_I = 53.0
##
###Mass fractions
##f_Cs = A_Cs/(A_Cs + A_I)
##f_I = A_I/(A_Cs + A_I)
##
###Nuclear properties for NaI
A_Na = 23.0
Z_Na = 11.0
##
f_Na = A_Na/(A_Na + A_I)
f_I1 = A_I/(A_Na + A_I)
##
###Nuclear properties for LiI
##A_Li = 6.0
##Z_Li = 3.0
##
##f_Li = A_Li/(A_Li + A_I)
##f_I2 = A_I/(A_Li + A_I)
##
###Nuclear properties for BaF2
##A_F = 19.0
##Z_F = 9.0
##A_Ba = 137.0
##Z_Ba = 56.0
##
##f_F = 2*A_F/(A_F + A_Ba)
##f_Ba = A_Ba/(A_F + A_Ba)
##
##
###Nuclear properties for PbWO4
##A_Pb = 208
##Z_Pb = 82
##A_W = 184
##Z_W = 74
##A_O = 16
##Z_O = 8
##
##f_Pb = A_Pb/(A_Pb+A_W+A_O)
##f_W = A_W/(A_Pb+A_W+A_O)
##f_O = 4*A_O/(A_Pb+A_W+A_O)
##
###Nuclear properties for Al2O3
A_Al = 27.0
Z_Al = 13.0
A_O = 16
Z_O = 8

f_Al = 2*A_Al/(A_Al+A_O)
f_O = 3*A_O/(A_Al+A_O)


for time in T_nu[ll:hl]:
   print("Now working with: ",time," s")

   ########################################################################
   ##                      Normalization for Supernova                   ##
   ########################################################################
   normalisation=1.0/(45.96e41) #1/(4 pi r^2) for betelgeuse

   flux_nue = diffFlux(E_nu, time, fnue_E, fnue_L)
   flux_anue = diffFlux(E_nu, time, fanue_E, fanue_L)
   flux_nux = diffFlux(E_nu, time, fnux_E, fnux_L)
   flux=(flux_nue + flux_anue + flux_nux*4)*normalisation
   neutrino_flux_list = InterpolatedUnivariateSpline(E_nu, flux, k = 1)
   Enu_min = np.min(E_nu)
   Enu_max = np.max(E_nu)

   neutrino_flux_Livermore = np.vectorize(neutrino_flux_list)

##   pl.plot(E_nu,flux,label=r'total')
##   pl.plot(E_nu,flux_nue,label=r'$\nu_e$')
##   pl.plot(E_nu,flux_nux,label=r'$\nu_x$')
##   pl.plot(E_nu,flux_anue,label=r'$\bar{\nu}_e$')
##   pl.legend( fontsize=14)
##   pl.show()
   

   norm = mass #* deltime_values[time_values.index(time)]
##   print(mass,norm,deltime_values[time_values.index(time)])

   ##
   ###COHERENT EVENT RATE vs Recoil Energy 
   ##
   ##print(1)
   ##
   ER_max_Ge = 300 ##upper cutoff in keV taken since 4 order of magnitude reduction in evt rate
   ER_max_Sapphire = 1000
   ##ER_max_Ge = 1000 ##upper cutoff in keV taken since 4 order of magnitude reduction in evt rate
   ##ER_max_Sapphire = 1000
   E_R1=np.logspace(-3.0, np.log10(ER_max_Ge), 51) ## bin the recoil energies upto upper cutoff ER_max
   E_R2=np.logspace(-3.0, np.log10(ER_max_Sapphire), 51)
   ##E_R1=np.linspace(0.001, ER_max_Ge, 201)
   ##E_R2=np.linspace(0.001, ER_max_Sapphire, 201)
   diffRate_CEvNS = np.vectorize(differentialRate_CEvNS)
   diffRate_full = np.vectorize(differentialRate_full)
   ##
   ##
   p1=lambda x:norm*diffRate_CEvNS(x, A_Si, Z_Si)
   p2=lambda x:norm*diffRate_CEvNS(x, A_Ge, Z_Ge)
   p3=lambda x:norm*(diffRate_CEvNS(x, A_Al, Z_Al)*f_Al+diffRate_CEvNS(x, A_O, Z_O)*f_O)
   ##p7=lambda x:norm*diffRate_CEvNS(x, A_I, Z_I)*f_I1
   ##p8=lambda x:norm*diffRate_CEvNS(x, A_Na, Z_Na)*f_Na


   ##CountI = np.zeros(len(E_R1)-1)
   ##CountNa = np.zeros(len(E_R1)-1)
   ##CountNaI = np.zeros(len(E_R1)-1)


   CountGe = np.zeros(len(E_R1)-1)
   #CountSi = np.zeros(len(E_R1)-1)
   CountAl2O3 = np.zeros(len(E_R2)-1)
   for i in range(0,len(E_R1)-1):
       print(i)
       CountGe[i]=quad(p2,E_R1[i],E_R1[i+1],epsrel=1e-4)[0]
       #CountSi[i]=quad(p1,E_R1[i],E_R1[i+1],epsrel=1e-4)[0]
       CountAl2O3[i]=quad(p3,E_R2[i],E_R2[i+1],epsrel=1e-4)[0]
   ##    CountI[i]=quad(p7,E_R1[i],E_R1[i+1],epsrel=1e-4)[0]
   ##    CountNa[i]=quad(p8,E_R1[i],E_R1[i+1],epsrel=1e-4)[0]
   ##    CountNaI[i]=CountI[i]+CountNa[i]

   print("Counts Ge: ", sum(CountGe))
   print("Counts Sapphire: ", sum(CountAl2O3))

   ##
   ##pl.xscale('log')
   ##pl.axvline(0.10, linestyle=':', color='red',linewidth=2)
   ##pl.axvline(0.010, linestyle=':', color='blue',linewidth=2)
   ####pl.loglog(E_R1, p1, label=r"Si")
   ######pl.loglog(E_R1, p2, label=r"Ge")
   ######pl.loglog(E_R1,p3,label=r"PbWO4")
   #######pl.loglog(E_R1,p4,label=r"Pb")
   #######pl.loglog(E_R1,p5,label=r"W")
   #######pl.loglog(E_R1,p6,label=r"O")
   ####pl.hist(E_R1[:-1], E_R1, weights=CountI,histtype='step', log=True,label=r'Iodine')
   ####pl.hist(E_R1[:-1], E_R1, weights=CountNa,histtype='step', log=True,label=r'Sodium')
   ####pl.hist(E_R1[:-1], E_R1, weights=CountNaI,histtype='step', log=True,label=r'Total')
   ##pl.hist(E_R1[:-1], E_R1, weights=CountGe,histtype='step', log=True,label=r'Ge',color='blue',linewidth=2)
   ###pl.hist(E_R1[:-1], E_R1, weights=CountSi,histtype='step', log=True,label=r'Si')
   ##pl.hist(E_R2[:-1], E_R2, weights=CountAl2O3,histtype='step', log=True,label=r'Sapphire',color='red',linewidth=2)
   ####pl.hist(evtErecGe, E_R1,histtype='step', log=True,label=r'Ge simulated')
   ######
   ########
   ######pl.ylim(0.2, 1e6)
   ##pl.yticks(fontsize=20)
   ##pl.xticks(fontsize=20)
   ##pl.xlim(0.001,ER_max_Sapphire)
   ##pl.xlabel("Recoil Energy (keV)",fontsize=24)
   ##pl.ylabel("events",fontsize=24)
   ##pl.legend( fontsize=18)
   ######pl.savefig("COHERENT_DiffDetectors.pdf", bbox_inches="tight")
   ##pl.show()

   ########################################################################
   ##             Define Ionization yeild and its derivative             ##
   ########################################################################
   def QL(E_R,Z=Z_Ge,A=A_Ge,mod="Lin"): #Lindhard Ionization yeild
       ## Lindhard parameters
       if(mod=="Lin"):
           k=0.133*(Z**(2.0/3.0))*(A**(-0.5))
           eps=11.5*(Z**(-7.0/3.0))*E_R
       ## J. Xu fitted parameters for NaI
       if(mod=="Xu"):
           k=0.0748267                   # y err only
           eps=0.00887421*E_R            # y err only
   ##      k=0.0748217                 # xy err
   ##      eps=0.00887428*E_R          # xy err
       ## H. W. Joo fitted parameters for yerr for NaI
       if(mod=="Joo"):
           if (abs(Z-11.0)<0.1):
               k=0.0446369
               eps=0.0201453*E_R
           elif(abs(Z-53.0)<0.1):
               k=0.0201308
               eps=0.00381294*E_R
       g  =3.0*(eps**0.15)+0.7*(eps**0.6)+eps
       return k*g/(1.0+k*g)


   def dQL(E_R,Z=Z_Ge,A=A_Ge,mod="Lin"): #derivative of YL
       if(mod=="Lin"):
           k=0.133*(Z**(2.0/3.0))*(A**(-0.5))
           eps=11.5*(Z**(-7.0/3.0))*E_R
       ## J. Xu fitted parameters for NaI
       if(mod=="Xu"):
           k=0.0748267                   # y err only
           eps=0.00887421*E_R            # y err only
   ##      k=0.0748217                 # xy err
   ##      eps=0.00887428*E_R          # xy err
       ## H. W. Joo fitted parameters for yerr for NaI
       if(mod=="Joo"):
           if (abs(Z-11.0)<0.1):
               k=0.0446369
               eps=0.0201453*E_R
           elif(abs(Z-53.0)<0.1):
               k=0.0201308
               eps=0.00381294*E_R
       g  =3.0*(eps**0.15)+0.7*(eps**0.6)+eps
       t1 =k/(1.0+k*g)-(g*k*k/(1.0+k*g)**2.0)
       t2 =0.45*(eps**(-0.85))+0.42*(eps**(-0.4))+1.0
       t3=eps/E_R
       return t1*t2*t3

   ## data from PHYS. REV. D 97, 022003 (2018)
   ## here the generated events are distributed into fiducial volume or veto volume
   ## the energy into charge is smeared with the given resolution
   ##QL = lambda Er: 0.16*Er**0.18 
   ##dQL = lambda Er: 0.0288*ER**(-0.82)
   sigEfid=0.2
   sigEveto=np.sqrt(2.0)*sigEfid
   #signormEheat=0.818/6.0

   def heatnorm(ER, V=69.0, Vfid=69.0): ## CDMSLite Run 
       return ER*(1.0+QL(ER)*V/3.0)/(1.0+Vfid/3.0)
      
   def signormEheat(E_R, sigE = 0.0127, B = 0.0008, A = 0.00549): ## From pg 136 of https://www.slac.stanford.edu/exp/cdms/ScienceResults/Theses/germond.pdf
   ##   Run2 Values from https://arxiv.org/pdf/1707.01632.pdf
      sigE = 0.00926
      B = 0.00064
      A = 5.68E-3
      E_Ree = heatnorm(E_R)
      #return 0.12
      return np.sqrt(sigE**2+B*E_Ree+(A*E_Ree)**2)



   EH = np.vectorize(heatnorm)

   ##sigESapphire = lambda er: 0.00078221*er*er-0.0118612*er+0.20143 #Fit from Nuclear Inst. and Methods in Physics Research, A 1046 (2023) 167634
   sigESapphire = lambda er: np.sqrt(0.00286518*er+0.025*0.025) #Fit from 2203.15903

   numruns=100
   numfinevts=np.zeros(numruns)
   numsimevtGe=np.zeros(numruns)
   numsimevtAl2O3=np.zeros(numruns)
   numevt10eV=np.zeros(numruns)
   numevt100eV=np.zeros(numruns)
   numevt170eV=np.zeros(numruns)
   numevt10eVnr=np.zeros(numruns)
   numevt100eVnr=np.zeros(numruns)
   numevt170eVnr=np.zeros(numruns)
   numevtSapphire=np.zeros(numruns)
   numevtSapphire50eV=np.zeros(numruns)
   evtnormEheatall= np.array([])
   evtsimall= np.array([])
   evtsimallsap= np.array([])

   def draw_from_hist(hist, bins, nsamples = 100000):
       cumsum = [0] + list(np.cumsum(hist))
       rand = np.random.rand(nsamples)*max(cumsum)
       return [np.interp(x, cumsum, bins) for x in rand]

   for run in range(numruns):
       ########################################################################
       ##   Generating recoil energy events in Ge as per the distribution    ##
       ########################################################################
       numevtsGe=np.random.poisson(sum(CountGe))   ## num of events to simulate assume a poisson statistics with mean given by the expected number
   ##    print("Run Number: ",run)
   ##    print("Number of events simulated in Ge: ", numevtsGe)
       numsimevtGe[run] = numevtsGe
       evtErecGe = draw_from_hist(CountGe, E_R1, numevtsGe)
       evtErecGe = np.array(evtErecGe)
       evtsimall=np.append(evtsimall,evtErecGe)

   ##    print("Num Sim Evt: ",len(evtErecGe))
   ##    
   ##    pl.hist(E_R1[:-1], bins=E_R1, weights=CountGe, histtype='step', log=True,label=r'Ge')
   ##    pl.hist(evtErecGe, bins=E_R1, histtype='step', log=True,label=r'Ge simulated')
   ##    pl.xlim(0.0,ER_max_Ge)
   ##    pl.xlabel("Recoil Energy (keV)")
   ##    pl.ylabel("events")
   ##    pl.legend( fontsize=14)
   ##    pl.show()
   ##    
       ########################################################################
       ##  Generating recoil energy events in Al2O3 as per the distribution  ##
       ########################################################################
       numevtsAl2O3=np.random.poisson(sum(CountAl2O3))   ## num of events to simulate assume a poisson statistics with mean given by the expected number
   ##    print("Run Number: ",run)
   ##    print("Number of events simulated in Sapphire: ", numevtsAl2O3)
       numsimevtAl2O3[run] = numevtsAl2O3
       evtErecAl2O3_noRes = draw_from_hist(CountAl2O3, E_R2, numevtsAl2O3)
       evtErecAl2O3 = [np.random.normal(er,sigESapphire(er)) for er in evtErecAl2O3_noRes]
       evtsimallsap=np.append(evtsimallsap,evtErecAl2O3)
       evtErecAl2O3=np.array(evtErecAl2O3)
       
       
   ##    evtErecAl2O3=np.zeros(numevtsAl2O3)
   ##
   ##    for i in range(numevtsAl2O3):       ## Generate the observed spectrum by MC simulation
   ##        acc=False                  ## flag to check if event accepted
   ##        #print(i)
   ##        count=0
   ##        while(acc==False):
   ##            count+=1
   ##            rn=np.random.rand(2)
   ##            er=rn[0]*ER_max_Sapphire
   ##            binnum=np.digitize(er,E_R2)-1
   ##            if(rn[1]<=CountAl2O3[binnum]/sum(CountAl2O3)):
   ##                #print(count,er)
   ##                acc=True
   ##                evtsimallsap=np.append(evtsimallsap,er)
   ##                evtErecAl2O3[i]=np.random.normal(er,sigESapphire(er)) ##here we also include the resolution in E_R
   ##
       numevtSapphire[run]=len(evtErecAl2O3[(evtErecAl2O3)>0.1])
       numevtSapphire50eV[run]=len(evtErecAl2O3[(evtErecAl2O3)>0.05])
   ##    print("number of events in Sapphire detector is: ", numevtSapphire[run])
       ########################################################################
       ##      Evaluating ionization and heat energies along with stdev      ##
       ########################################################################
       evtErecGefid = np.array([])
       evtErecGeveto = np.array([])
       Efid = np.array([])
       Eveto = np.array([])
       #total normalized thermal energy includes thermal energy due to recoil and NTL phonons
       #normalized for normEheat=Efid=Er for electron recoils
       evtnormEheat= np.array([])
       Eheat= np.array([])
       evtnormEheatfid = np.array([])
       evtnormEheatveto = np.array([])
       for evt in evtErecGe:
           sigERee = signormEheat(evt)
           fn=lambda x: heatnorm(x)-sigERee
           sigER = fsolve(fn,sigERee)
           Eheat = np.append(Eheat,np.random.normal(evt,sigER))
           if(np.random.rand()<0.75):
               evtErecGefid=np.append(evtErecGefid,evt)              #events in fiducial volume
               efid=QL(evt)*evt
               evet=0.0
               Efid=np.append(Efid,np.random.normal(efid,sigEfid))
               Eveto=np.append(Eveto,np.random.normal(evet,sigEveto))
               normEheat=np.random.normal(heatnorm(evt),signormEheat(evt))
               evtnormEheat=np.append(evtnormEheat,normEheat)
               evtnormEheatall=np.append(evtnormEheatall,normEheat)
               evtnormEheatfid=np.append(evtnormEheatfid,normEheat)
               #evtnormEheatveto=np.append(evtnormEheatveto,0.0)
               
           else:
               evtErecGeveto=np.append(evtErecGeveto,evt)             #events in veto volume
               efid=QL(evt)*evt*0.5
               evet=QL(evt)*evt
               Efid=np.append(Efid,np.random.normal(efid,sigEfid))
               Eveto=np.append(Eveto,np.random.normal(evet,sigEveto))
               normEheat=np.random.normal(heatnorm(evt),signormEheat(evt))
               evtnormEheat=np.append(evtnormEheat,normEheat)
               evtnormEheatall=np.append(evtnormEheatall,normEheat)
               evtnormEheatveto=np.append(evtnormEheatveto,normEheat)
               #evtnormEheatfid=np.append(evtnormEheatfid,0.0)

          
       ########################################################################
       ##            Applying fiducial cuts and printing results             ##
       ########################################################################
       
   ##    print("number of fiducial events: ", len(evtnormEheatfid))
   ##    print("number of surface events: ", len(evtnormEheatveto))
       numevt10eV[run]=len(evtnormEheat[(evtnormEheat)>0.01])
       numevt100eV[run]=len(evtnormEheat[(evtnormEheat)>0.1])
       numevt170eV[run]=len(evtnormEheat[(evtnormEheat)>0.17])
       numevt10eVnr[run]=len(Eheat[(Eheat)>0.01])
       numevt100eVnr[run]=len(Eheat[(Eheat)>0.1])
       numevt170eVnr[run]=len(Eheat[(Eheat)>0.17])

   avgNumEvtGe10eVee[time] = np.average(numevt10eV)
   avgNumEvtGe100eVee[time] = np.average(numevt100eV)
   avgNumEvtGe170eVee[time] = np.average(numevt170eV)

   avgNumEvtGe10eVth[time] = np.average(numevt10eVnr)
   avgNumEvtGe100eVth[time] = np.average(numevt100eVnr)
   avgNumEvtGe170eVth[time] = np.average(numevt170eVnr)

   stdevNumEvtGe10eVee[time] = np.std(numevt10eV)
   stdevNumEvtGe100eVee[time] = np.std(numevt100eV)
   stdevNumEvtGe170eVee[time] = np.std(numevt170eV)

   stdevNumEvtGe10eVth[time] = np.std(numevt10eVnr)
   stdevNumEvtGe100eVth[time] = np.std(numevt100eVnr)
   stdevNumEvtGe170eVth[time] = np.std(numevt170eVnr)

   avgNumEvtSap50eVth[time] = np.average(numevtSapphire50eV)
   avgNumEvtSap100eVth[time] = np.average(numevtSapphire)

   stdevNumEvtSap50eVth[time] = np.std(numevtSapphire50eV)
   stdevNumEvtSap100eVth[time] = np.std(numevtSapphire)

writefnameGe = "SNn_Counts_Ge"+str(T_nu[ll])+"s-"+str(T_nu[hl-1])+"s_v1.dat"
wf = open("data/intp3003_Ge/Livermore/"+writefnameGe,'wt')
line0 = "time \t delTime \t\t mean eVee \t \t stdev eVee \t \t mean eVth \t \t stdev eVth \n"
line1 = " \t  \t 10 \t 100 \t 170 \t 10 \t 100 \t 170 \t 10 \t 100 \t 170 \t 10 \t 100 \t 170 \n"
wf.writelines(line0)
wf.writelines(line1)

for key in avgNumEvtGe10eVee.keys():
    line2 = str(key)+" \t "+str(deltime_values[np.where(T_nu==key)[0][0]])+" \t "+str(avgNumEvtGe10eVee[key])+" \t "+str(avgNumEvtGe100eVee[key])+" \t "+str(avgNumEvtGe170eVee[key])+" \t "+str(stdevNumEvtGe10eVee[key])+" \t "+str(stdevNumEvtGe100eVee[key])+" \t "+str(stdevNumEvtGe170eVee[key])
    line2 = line2 + " \t "+str(avgNumEvtGe10eVth[key])+" \t "+str(avgNumEvtGe100eVth[key])+" \t "+str(avgNumEvtGe170eVth[key])+" \t "+str(stdevNumEvtGe10eVth[key])+" \t "+str(stdevNumEvtGe100eVth[key])+" \t "+str(stdevNumEvtGe170eVth[key])+"\n"
    wf.writelines(line2)
wf.close()

writefnameSap = "SNn_Counts_Sap"+str(T_nu[ll])+"s-"+str(T_nu[hl-1])+"s_v1.dat"
wf = open("data/intp3003_Sap/Livermore/"+writefnameSap,'wt')
line0 = "time  \t \t mean eVth \t \t stdev eVth \n"
line1 = " \t 50 \t 100 \t 50 \t 100 \n"
wf.writelines(line0)
wf.writelines(line1)

for key in avgNumEvtGe10eVee.keys():
    line2 = str(key)+" \t "+str(deltime_values[np.where(T_nu==key)[0][0]])+" \t "+str(avgNumEvtSap50eVth[key])+" \t "+str(avgNumEvtSap100eVth[key])+" \t "+str(stdevNumEvtSap50eVth[key])+" \t "+str(stdevNumEvtSap100eVth[key])+"\n"
    wf.writelines(line2)
wf.close()

##    if(run%100==0):
##    print("number of events in Ge with normalised recoil energy > 010eV is: ", numevt10eV[run])
##    print("number of events in Ge with normalised recoil energy > 100eV is: ", numevt100eV[run])
##    print("number of events in Ge with normalised recoil energy > 170eV is: ", numevt170eV[run])
##
##    print("number of events in Ge with recoil energy > 010eV is: ", numevt10eVnr[run])
##    print("number of events in Ge with recoil energy > 0100eV is: ", numevt100eVnr[run])
##    print("number of events in Ge with recoil energy > 0170eV is: ", numevt170eVnr[run])
##
##print("average simulated counts for 50eV threshold Sapphire: ",np.average(numevtSapphire50eV))
##print("average simulated counts for 100eV threshold Sapphire: ",np.average(numevtSapphire))
##print("average simulated counts for 10eVee threshold Ge: ",np.average(numevt10eV))
##print("average simulated counts for 100eVee threshold Ge: ",np.average(numevt100eV))
##print("average simulated counts for 170eVee threshold Ge: ",np.average(numevt170eV))
##print("average simulated counts for 10eVnr threshold Ge: ",np.average(numevt10eVnr))
##print("average simulated counts for 100eVnr threshold Ge: ",np.average(numevt100eVnr))
##print("average simulated counts for 170eVnr threshold Ge: ",np.average(numevt170eVnr))
##pl.xscale('log')
##pl.hist(E_R1[:-1], E_R1, weights=numruns*CountGe,histtype='step', log=True,label=r'Ge_Expected',color='blue',linewidth=2)
##pl.hist(evtsimall,bins=E_R1,histtype='step', log=True,label=r'Ge_Simulated',color='red',linewidth=2)
##pl.yticks(fontsize=20)
##pl.xticks(fontsize=20)
##pl.xlim(0.001,ER_max_Sapphire)
##pl.xlabel("Recoil Energy (keV)",fontsize=24)
##pl.ylabel("events",fontsize=24)
##pl.legend( fontsize=18)
##pl.show()
##
##pl.xscale('log')
##pl.hist(E_R2[:-1], E_R2, weights=numruns*CountAl2O3,histtype='step', log=True,label=r'Sapphire',color='green',linewidth=2)
##pl.hist(evtsimallsap,bins=E_R2,histtype='step', log=True,label=r'Sapphire_Simulated',color='black',linewidth=2)
##pl.yticks(fontsize=20)
##pl.xticks(fontsize=20)
##pl.xlim(0.001,ER_max_Sapphire)
##pl.xlabel("Recoil Energy (keV)",fontsize=24)
##pl.ylabel("events",fontsize=24)
##pl.legend( fontsize=18)
##pl.show()

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

###########################################################################
##                       Getting the time values                         ##
###########################################################################
folder_path = "data/intp2001/"  
time_values = []

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file
for file_name in files:
    # Split the filename by underscore and dot to extract the time value
    parts = file_name.split('_')
    if len(parts) == 2:
        time_value = parts[1].split('.d')[0]  # Remove the extension
##        print(time_value,parts[1].split('.d')[1])
        time_values.append(time_value)
time_values.sort(key=float)
##print(time_values)

deltime_values = np.array([])
for i in range(len(time_values)):
    if(i==0):
        deltime = float(time_values[i+1])-float(time_values[i])
    elif(i==len(time_values)-1):
        deltime = float(time_values[i])-float(time_values[i-1])
    else:
        deltime = 0.5*(float(time_values[i+1])-float(time_values[i-1]))
    deltime_values=np.append(deltime_values,deltime)
############################################################################
ll = 75
hl = 76

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

CountGe10eVth = {}
CountGe50eVth = {}
CountGe100eVth= {}

CountSap100eVth= {}

avgNumEvtGe10eVee = {}
avgNumEvtGe100eVee = {}
avgNumEvtGe50eVee = {}

avgNumEvtGe10eVth = {}
avgNumEvtGe100eVth = {}
avgNumEvtGe50eVth = {}

stdevNumEvtGe10eVee = {}
stdevNumEvtGe100eVee = {}
stdevNumEvtGe50eVee = {}

stdevNumEvtGe10eVth = {}
stdevNumEvtGe100eVth = {}
stdevNumEvtGe50eVth = {}

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
                        *neutrino_flux_nakazato(E_nu)
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


for time in time_values[ll:hl]:
   print(time," : Now Loading : data/intp2001/intp2001_"+time+".dat" )
   ##data = np.loadtxt("SNn_TimeIntegrated_"+SNmass+"M"+revTime+".dat") #MeV vs No./MeV
   data = np.loadtxt("data/intp2001/intp2001_"+time+".dat") #MeV vs No./MeV
   ########################################################################
   ##                       Normalization for Reactor                    ##
   ########################################################################
   ##ReactPower = 2.815  #in GW thermal
   ##AvEperFission = 205 #in MeV
   ##FissperDay = ReactPower/ee*1E3/AvEperFission*86400.0
   ##dist = 23.7 #in m
   ##normalisation=FissperDay/(4*np.pi*dist*dist*100*100)

   ########################################################################
   ##                      Normalization for Supernova                   ##
   ########################################################################
   normalisation=1.0/(45.96e41) #1/(4 pi r^2) for betelgeuse

   ## Note that for supernovas, the spectrum is already in histogram format
   ## i.e. as counts/MeV instead of dN/dE. thus multiplyingby binwidth will
   ## directly give counts in that bin

   neutrino_flux_list = InterpolatedUnivariateSpline(np.sqrt(data[:,0]*data[:,1]), (data[:,2]+data[:,3]+data[:,4]*4)*normalisation, k = 1) #MeV vs 1/MeV/m^2
   bins=np.append(data[:,0],data[-1,1])
   flux=(data[:,2]+data[:,3]+data[:,4]*4)*normalisation
   Enu_min = np.min(data[:,0])
   Enu_max = np.max(data[:,1])

   #Now tabulate the total neutrino flux
   Evals = np.logspace(np.log10(Enu_min), np.log10(Enu_max), 100)
   flux_tab = 0.0*Evals
   flux_tab += neutrino_flux_list(Evals)
   #neutrino_flux_nakazato = InterpolatedUnivariateSpline(Evals,flux_tab, k = 1)
   flux_tab[-1]=0.0

   def neutrino_flux_hist(Enu,b=bins,f=flux):
       #print(Enu)
       diffar=b-Enu
       arg=np.abs(diffar).argmin()
   ##    return 1e15
       if(diffar[arg]>=0.0):
           return f[arg-1]
       elif(diffar[arg]<0.0):
           return f[arg]

   neutrino_flux_nakazato=np.vectorize(neutrino_flux_hist)

   ###Plot neutrino flux
##   pl.hist(bins[:-1], bins, weights=data[:,2],histtype='step',label=r'$\nu_e$',color='blue',linewidth=2)
##   pl.hist(bins[:-1], bins, weights=data[:,3],histtype='step',label=r'$\bar{\nu}_e$',color='green',linewidth=2)
##   pl.hist(bins[:-1], bins, weights=data[:,4],histtype='step',label=r'$\nu_x$',color='red',linewidth=2)
##   pl.xlim(0,50)
##   pl.xlabel("neutrino energy [MeV]")
##   pl.ylabel("total no. of neutrino [MeV$^{-1}$]")
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
   E_R1=np.logspace(-3.0, np.log10(ER_max_Ge), 201) ## bin the recoil energies upto upper cutoff ER_max
   E_R2=np.logspace(-3.0, np.log10(ER_max_Sapphire), 201)
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

   CountGe10eVth[time] = sum(CountGe[np.where(E_R1>0.01)[0][0]:])
   CountGe50eVth[time] = sum(CountGe[np.where(E_R1>0.05)[0][0]:])
   CountGe100eVth[time] = sum(CountGe[np.where(E_R1>0.1)[0][0]:])

   CountSap100eVth[time] = sum(CountAl2O3[np.where(E_R2>0.1)[0][0]:])

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
           k=0.133*(Z**(2.0/3.0))*(A**(0.5))
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
           k=0.133*(Z**(2.0/3.0))*(A**(0.5))
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

   numruns=300
   numfinevts=np.zeros(numruns)
   numsimevtGe=np.zeros(numruns)
   numsimevtAl2O3=np.zeros(numruns)
   numevt10eV=np.zeros(numruns)
   numevt100eV=np.zeros(numruns)
   numevt50eV=np.zeros(numruns)
   numevt10eVnr=np.zeros(numruns)
   numevt100eVnr=np.zeros(numruns)
   numevt50eVnr=np.zeros(numruns)
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
       print("Run Number: ",run)
       print("Number of events simulated in Ge: ", numevtsGe)
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
       numevt50eV[run]=len(evtnormEheat[(evtnormEheat)>0.17])
       numevt10eVnr[run]=len(Eheat[(Eheat)>0.01])
       numevt100eVnr[run]=len(Eheat[(Eheat)>0.1])
       numevt50eVnr[run]=len(Eheat[(Eheat)>0.17])

   avgNumEvtGe10eVee[time] = np.average(numevt10eV)
   avgNumEvtGe100eVee[time] = np.average(numevt100eV)
   avgNumEvtGe50eVee[time] = np.average(numevt50eV)

   avgNumEvtGe10eVth[time] = np.average(numevt10eVnr)
   avgNumEvtGe100eVth[time] = np.average(numevt100eVnr)
   avgNumEvtGe50eVth[time] = np.average(numevt50eVnr)

   stdevNumEvtGe10eVee[time] = np.std(numevt10eV)
   stdevNumEvtGe100eVee[time] = np.std(numevt100eV)
   stdevNumEvtGe50eVee[time] = np.std(numevt50eV)

   stdevNumEvtGe10eVth[time] = np.std(numevt10eVnr)
   stdevNumEvtGe100eVth[time] = np.std(numevt100eVnr)
   stdevNumEvtGe50eVth[time] = np.std(numevt50eVnr)

   avgNumEvtSap50eVth[time] = np.average(numevtSapphire50eV)
   avgNumEvtSap100eVth[time] = np.average(numevtSapphire)

   stdevNumEvtSap50eVth[time] = np.std(numevtSapphire50eV)
   stdevNumEvtSap100eVth[time] = np.std(numevtSapphire)

writefnameGe = "SNn_Counts_Ge"+time_values[ll]+"s-"+time_values[hl-1]+"s.dat"
wf = open("data/intp2001_Ge/"+writefnameGe,'wt')
line0 = "time \t delTime \t\t mean eVee \t \t stdev eVee \t \t mean eVth \t \t stdev eVth \t \t CountGe \n"
line1 = " \t  \t 10 \t 100 \t 50 \t 10 \t 100 \t 50 \t 10 \t 100 \t 50 \t 10 \t 100 \t 50 \t 10 \t 100 \t 50 \n"
wf.writelines(line0)
wf.writelines(line1)

for key in avgNumEvtGe10eVee.keys():
    line2 = str(key)+" \t "+str(deltime_values[time_values.index(key)])+" \t "+str(avgNumEvtGe10eVee[key])+" \t "+str(avgNumEvtGe100eVee[key])+" \t "+str(avgNumEvtGe50eVee[key])+" \t "+str(stdevNumEvtGe10eVee[key])+" \t "+str(stdevNumEvtGe100eVee[key])+" \t "+str(stdevNumEvtGe50eVee[key])
    line2 = line2 + " \t "+str(avgNumEvtGe10eVth[key])+" \t "+str(avgNumEvtGe100eVth[key])+" \t "+str(avgNumEvtGe50eVth[key])+" \t "+str(stdevNumEvtGe10eVth[key])+" \t "+str(stdevNumEvtGe100eVth[key])+" \t "+str(stdevNumEvtGe50eVth[key])
    line2 = line2 + "\t"+ str(CountGe10eVth[key])+ "\t"+ str(CountGe100eVth[key])+ "\t"+ str(CountGe50eVth[key])+"\n"
    wf.writelines(line2)
wf.close()

writefnameSap = "SNn_Counts_Sap"+time_values[ll]+"s-"+time_values[hl-1]+"s.dat"
wf = open("data/intp2001_Sap/"+writefnameSap,'wt')
line0 = "time  \t \t mean eVth \t \t stdev eVth \n"
line1 = " \t 50 \t 100 \t 50 \t 100 \n"
wf.writelines(line0)
wf.writelines(line1)

for key in avgNumEvtGe10eVee.keys():
    line2 = str(key)+" \t "+str(deltime_values[time_values.index(key)])+" \t "+str(avgNumEvtSap50eVth[key])+" \t "+str(avgNumEvtSap100eVth[key])+" \t "+str(stdevNumEvtSap50eVth[key])+" \t "+str(stdevNumEvtSap100eVth[key])+"\n"
    wf.writelines(line2)
wf.close()

##    if(run%100==0):
##    print("number of events in Ge with normalised recoil energy > 010eV is: ", numevt10eV[run])
##    print("number of events in Ge with normalised recoil energy > 100eV is: ", numevt100eV[run])
##    print("number of events in Ge with normalised recoil energy > 50eV is: ", numevt50eV[run])
##
##    print("number of events in Ge with recoil energy > 010eV is: ", numevt10eVnr[run])
##    print("number of events in Ge with recoil energy > 0100eV is: ", numevt100eVnr[run])
##    print("number of events in Ge with recoil energy > 050eV is: ", numevt50eVnr[run])
##
##print("average simulated counts for 50eV threshold Sapphire: ",np.average(numevtSapphire50eV))
##print("average simulated counts for 100eV threshold Sapphire: ",np.average(numevtSapphire))
##print("average simulated counts for 10eVee threshold Ge: ",np.average(numevt10eV))
##print("average simulated counts for 100eVee threshold Ge: ",np.average(numevt100eV))
##print("average simulated counts for 50eVee threshold Ge: ",np.average(numevt50eV))
##print("average simulated counts for 10eVnr threshold Ge: ",np.average(numevt10eVnr))
##print("average simulated counts for 100eVnr threshold Ge: ",np.average(numevt100eVnr))
##print("average simulated counts for 50eVnr threshold Ge: ",np.average(numevt50eVnr))
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

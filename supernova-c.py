# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl

from scipy.integrate import quad
from scipy.interpolate import interp1d, UnivariateSpline,InterpolatedUnivariateSpline

from scipy.optimize import minimize

#from tqdm import tqdm

#Change default font size so you don't need a magnifying glass
matplotlib.rc('font', **{'size'   : 16})

#import CEvNS
#help(CEvNS.xsec_CEvNS)

#----Constants----
G_FERMI = 1.1664e-5     #Fermi Constant in GeV^-2
SIN2THETAW = 0.2387     #Sine-squared of the weak angle
ALPHA_EM = 0.007297353  #EM Fine structure constant
m_e = 0.5109989461e-3   #Electron mass in GeV  
SQRT2 = np.sqrt(2.0)  

#----Module-scope variables----
neutrino_flux_tot = None
neutrino_flux_list = None
Enu_min = 0
Enu_max = 0


#Spectra is in 1/MeV
neutrino_flux_list = [lambda x: 1e-30 for i in range(6)]
data = np.loadtxt("supernova-spectrum.txt")
normalisation=1.0/(45.96e41)
neutrino_flux_list = InterpolatedUnivariateSpline(np.sqrt(data[:,0]*data[:,0]), (data[:,2]+data[:,3]+data[:,4]*4)*normalisation, k = 1)
Enu_min = np.min(data[:,0])
Enu_max = np.max(data[:,1])


#Now tabulate the total neutrino flux
Evals = np.logspace(np.log10(Enu_min), np.log10(Enu_max), 100)
flux_tab = 0.0*Evals
flux_tab += neutrino_flux_list(Evals)
neutrino_flux_tot = InterpolatedUnivariateSpline(Evals,flux_tab, k = 1)
flux_tab[-1]=0.0
print("Total Neutrino Flux: ", neutrino_flux_tot)

#Plot neutrino flux
E_nu = np.logspace(0, np.log10(300),1000)
E_nu = np.linspace(0, 300,1000)

pl.figure()
pl.ylim(1e5, 1e15)
pl.semilogy(E_nu, neutrino_flux_tot(E_nu))
#pl.plot(E_nu, neutrino_flux_tot(E_nu)/1e12)


#pl.ylim(-0.1,10)
pl.title(r"Neutrino flux at SuperNova", fontsize=12)
pl.xlabel(r"Neutrino energy, $E_\nu$ [MeV] $")
pl.ylabel(r"$\Phi_\nu$ [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
pl.show()

def HelmFormFactor(E, A):#requires E in keV
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
    return (F*F)*(np.exp(-(q2*q2*s*s)))

#Maximum nuclear recoil energy (in keV)
def ERmax(E_nu, A):
    #Nuclear mass in MeV
    m_A_MeV = A*0.9315e3
    #return 1e3*2.0*E_nu**2/m_N_MeV
    return 1e3*(2.0*E_nu*E_nu)/(m_A_MeV + 2*E_nu)

def xsec_CEvNS(E_R, E_nu, A, Z, gsq=0.0, m_med=1000.0):
    m_A = A*0.9315 #Mass of target nucleus (in GeV)
    q = np.sqrt(2.0*E_R*m_A) #Recoil momentum (in MeV)
    #Note: m_A in GeV, E_R in keV, E_nu in MeV, m_med in MeV
    
    #Calculate SM contribution
    Qv = (A-Z) - (1.0-4.0*SIN2THETAW)*Z #Coherence factor
    xsec_SM = (G_FERMI*G_FERMI/(4.0*np.pi))*Qv*Qv*m_A*   \
        (1.0-(q*q)/(4.0*E_nu*E_nu)) #in GeV^-3
    
    #Calculate New-Physics correction from Z' coupling
    #Assume universal coupling to quarks (u and d)
    QvNP = 3.0*A*gsq

    #Factor of 1e6 from (GeV/MeV)^2
    G_V = 1 - 1e6*(SQRT2/G_FERMI)*(QvNP/Qv)*1.0/(q*q + m_med*m_med)
    
    #Convert from (GeV^-3) to (cm^2/keV)
    #and multiply by form factor and New Physics correction
    return G_V*G_V*xsec_SM*1e-6*(1.97e-14)*(1.97e-14)*HelmFormFactor(E_R, A)

##def xsec_CEvNS(E_R, E_nu, A, Z, gsq=0.0, m_med=1000.0):
##    m_A = A*0.9315 #Mass of target nucleus (in GeV)
##    Qv = (A-Z) - (1.0-4.0*SIN2THETAW)*Z #Coherence factor
##    Efr=(E_R/E_nu)
##    xsec=(G_FERMI**2)/(8*np.pi)*(Qv**2)*m_A*(Efr**2-2*Efr+2-m_A*Efr/E_nu) #GeV^-3
##    return xsec*(1.973e-14*1.973e-14)*1e-6 #cm^2/keV


def differentialRate_CEvNS(E_R, A, Z, gsq=0.0, m_med=1000.0):
    integrand = lambda E_nu: xsec_CEvNS(E_R, E_nu, A, Z, gsq, m_med)\
                        *neutrino_flux_tot(E_nu)
    E_min = np.sqrt(A*0.9315*E_R/2) #(in MeV)
    E_min = np.maximum(E_min, Enu_min)
    #For reactor neutrinos, set E_max:
    E_max = Enu_max
    if (E_min > E_max):
        return 0
    m_N = A*1.66054e-27 #Nucleus mass in kg
    rate = quad(integrand, E_min, E_max, epsrel=1e-4)[0]/m_N
    return rate #Convert from (per second) to (per day)

def differentialRate_full(E_R, A, Z, gsq=0.0, m_med=1000.0, mu_nu=0.0):
    return differentialRate_CEvNS(E_R, A, Z, gsq, m_med)



#COHERENT EVENT RATE
#Nuclear properties for Si
A_Si = 28.0
Z_Si = 14.0

#Nuclear properties for Ge
A_Ge = 73.0
Z_Ge = 32.0

#Nuclear properties for Ge
A_Ar = 40.0
Z_Ar = 18.0


#Nuclear properties for CsI
A_Cs = 133.0
Z_Cs = 55.0
A_I = 127.0
Z_I = 53.0

#Mass fractions
f_Cs = A_Cs/(A_Cs + A_I)
f_I = A_I/(A_Cs + A_I)

#Nuclear properties for NaI
A_Na = 23.0
Z_Na = 11.0

f_Na = A_Na/(A_Na + A_I)
f_I1 = A_I/(A_Na + A_I)

#Nuclear properties for LiI
A_Li = 6.0
Z_Li = 3.0

f_Li = A_Li/(A_Li + A_I)
f_I2 = A_I/(A_Li + A_I)

#Nuclear properties for BaF2
A_F = 19.0
Z_F = 9.0
A_Ba = 137.0
Z_Ba = 56.0

f_F = 2*A_F/(A_F + A_Ba)
f_Ba = A_Ba/(A_F + A_Ba)


A_Pb = 208
Z_Pb = 82
A_W = 184
Z_W = 74
A_O = 16
Z_O = 8

f_Pb = A_Pb/(A_Pb+A_W+A_O)
f_W = A_W/(A_Pb+A_W+A_O)
f_O = 4*A_O/(A_Pb+A_W+A_O)

mass = 30.0 #target mass in kg
time = 1.0 #exposure time in days

dist = 1.0
Pth=1.0
#norm=(Pth/3950.)*(30.0**2)/(dist**2)
norm=1.0
# Normalised for the flux https://doi.org/10.1007/JHEP04(2020)054 CONNIE 
# Count rates reproduced for Miner NIMA 853 (2017) 53

#COHERENT EVENT RATE vs Recoil Energy 
weight=1.0

print(1)

E_R1=np.linspace(0.0001, 50., 100)
diffRate_CEvNS = np.vectorize(differentialRate_CEvNS)
diffRate_full = np.vectorize(differentialRate_full)


p1=mass*diffRate_full(E_R1, A_Si, Z_Si, mu_nu=2.2e-12)
p2=mass*diffRate_CEvNS(E_R1, A_Ge, Z_Ge)
p3=mass*(f_Pb*diffRate_CEvNS(E_R1, A_Pb, Z_Pb)+f_W*diffRate_CEvNS(E_R1, A_W, Z_W)+f_O*diffRate_CEvNS(E_R1, A_O, Z_O))
p4=mass*diffRate_CEvNS(E_R1, A_Pb, Z_Pb)
p5=mass*diffRate_CEvNS(E_R1, A_W, Z_W)
p6=mass*diffRate_CEvNS(E_R1, A_O, Z_O)



pl.axvline(0.10, linestyle=':', color='k')
pl.semilogy(E_R1, p1, label=r"Si")
pl.loglog(E_R1, p2, label=r"Ge")
pl.loglog(E_R1,p3,label=r"PbWO4")
#pl.loglog(E_R1,p4,label=r"Pb")
#pl.loglog(E_R1,p5,label=r"W")
#pl.loglog(E_R1,p6,label=r"O")



pl.ylim(1e-1, 1e5)
pl.xlim(0.01,50)
pl.xlabel("Recoil Energy (keV)")
pl.ylabel("[events/30kg/keV/explosion]")
pl.legend( fontsize=14)
pl.savefig("COHERENT_DiffDetectors.pdf", bbox_inches="tight")
pl.show()

print(2)
#COHERENT EVENT RATE vs Recoil Energy 
#COHERENT EVENT RATE vs Recoil Energy 

E_R=np.logspace(-3,2,1000)

print(3)

diffRate_CEvNS = np.vectorize(differentialRate_CEvNS)
diffRate_full = np.vectorize(differentialRate_full)

print(4)

##pl.loglog(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=3.0e-10)/1e6, label=r"$\mu_{\nu} / \mu_B = 3.0 X 10^{-10}$")
##pl.loglog(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=1.0e-10)/1e6, label=r"$\mu_{\nu} / \mu_B = 1.0 X 10^{-10}$")
##pl.loglog(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=5.0e-11)/1e6, label=r"$\mu_{\nu} / \mu_B = 5.0 X 10^{-11}$")
##pl.loglog(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=1.0e-12)/1e6, label=r"$\mu_{\nu} / \mu_B = 1.0 X 10^{-12}$")
##pl.axvline(0.1, linestyle='--', color='k')
##pl.ylim(1e-5, 1e4)
##pl.xlim(1e-2, 5)
##pl.xlabel("Recoil Energy (keV)")
##pl.ylabel("[events/10kg/keV/yr] x $10^6$")
##pl.legend( fontsize=12)
##pl.savefig("COHERENT_magnetic-mom.pdf", bbox_inches="tight")
##pl.show()


dRdPE_Si = lambda x: mass*norm*time*diffRate_CEvNS(x, A_Si, Z_Si)
dRdPE_Ge = lambda x: mass*norm*time*diffRate_CEvNS(x, A_Ge, Z_Ge)
dRdPE_Ar = lambda x: mass*norm*time*diffRate_CEvNS(x, A_Ar, Z_Ar)
##dRdPE_PbWO4 = lambda x: mass*norm*time*(f_Pb*diffRate_CEvNS(x, A_Pb, Z_Pb)+f_W*diffRate_CEvNS(x, A_W, Z_W)+f_O*diffRate_CEvNS(x, A_O, Z_O))
PE_bins = np.linspace(5,10,101)

N_SM_Ge = np.zeros(1000)
N_SM_Si = np.zeros(1000)
##N_SM_PbWO4 = np.zeros(1000)

for i in range(100):
    N_SM_Ge[i] = quad(dRdPE_Ge, PE_bins[i], PE_bins[i+1], epsabs = 0.01)[0]
    N_SM_Si[i] = quad(dRdPE_Si, PE_bins[i], PE_bins[i+1], epsabs = 0.01)[0]
    ##N_SM_PbWO4[i] = quad(dRdPE_PbWO4, PE_bins[i], PE_bins[i+1], epsabs = 0.01)[0]

##pl.step(PE_bins, np.append(N_SM_tot,0), 'g', linestyle="-", where = "post", label="CEvNS signal (this work)",linewidth=1.5)
##pl.axhline(0, linestyle='--', color = 'gray')
##pl.xlabel("Number of photoelectrons (PE)")
##pl.ylabel("Expected Counts")
##pl.legend( fontsize=14)
##pl.xlim(0, 10)
##pl.show()

print("Total CEvNS events expected Ge: ", np.sum(N_SM_Ge))
print("Total CEvNS events expected Si: ", np.sum(N_SM_Si))
##print("Total CEvNS events expected PbWO4: ", np.sum(N_SM_PbWO4))


def IonYeild(E_R):
    return 0.16*E_R**0.18 ##E_R in keV

def E_Ion(E_R):
    return IonYeild(E_R)*E_R ##E_R in keV

def E_Heat(E_R,V):
    numr=1+IonYeild(E_R)*V/2.96e-3 #E_R and V in keV
    denm=1+V/2.96e-3
    return E_R*numr/denm

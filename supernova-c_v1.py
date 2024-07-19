# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pl
from array import *

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
neutrino_flux_nakazato = None
neutrino_flux_list = None
Enu_min = 0
Enu_max = 0


neutrino_flux_list = [lambda x: 1e-30 for i in range(6)]
fname="data/supernova-spectrum_brightest.txt"
data = np.loadtxt(fname)
print(fname)
normalisation=1.0/(45.96e41)#
neutrino_flux_list = InterpolatedUnivariateSpline(np.sqrt(data[:,0]*data[:,1]), (data[:,2]+data[:,3]+data[:,4]*4)*normalisation, k = 1)
bins=np.append(data[:,0],data[-1,1])
flux=(data[:,2]+data[:,3]+data[:,4]*4)*normalisation
Enu_min = np.min(data[:,0])
Enu_max = np.max(data[:,1])

#Now tabulate the total neutrino flux
Evals = np.logspace(np.log10(Enu_min), np.log10(Enu_max), 100)
flux_tab = 0.0*Evals
flux_tab += neutrino_flux_list(Evals)
neutrino_flux_nakazato = InterpolatedUnivariateSpline(Evals,flux_tab, k = 1)
flux_tab[-1]=0.0
print("Total Neutrino Flux: ", neutrino_flux_nakazato)

def neutrino_flux_hist(Enu,b=bins,f=flux):
    diffar=b-Enu
    arg=np.abs(diffar).argmin()
    if(diffar[arg]>=0.0):
        return f[arg-1]
    elif(diffar[arg]<0.0):
        return f[arg]
##neutrino_flux_nakazato=np.vectorize(neutrino_flux_hist)

pl.hist(bins[:-1], bins, weights=data[:,2],histtype='step',label=r'$\nu_e$')
pl.hist(bins[:-1], bins, weights=data[:,3],histtype='step',label=r'$\bar{\nu}_e$')
pl.hist(bins[:-1], bins, weights=data[:,4],histtype='step',label=r'$\nu_x$')
pl.xlim(0,50)
pl.xlabel("neutrino energy [MeV]")
pl.ylabel("total no. of neutrino [MeV$^{-1}$]")
pl.legend( fontsize=14)
pl.show()

#Plot neutrino flux
#E_nu = np.logspace(0, np.log10(300),1000)
E_nu = np.linspace(0, 300,1000)
print(0)
pl.figure()
pl.xlim(0,60)
pl.ylim(0,1e57*normalisation)
#pl.semilogy(E_nu, neutrino_flux_nakazato(E_nu))
#
###pl.plot(E_nu, neutrino_flux_tot(E_nu)/1e12)
##
##
##pl.ylim(0,0.25e15)
##pl.title(r"Neutrino flux at Reactor", fontsize=12)
pl.xlabel(r"Neutrino energy, $E_\nu$ [MeV] $")
pl.ylabel(r"$\nu$ [$MeV$^{-1}$]")
pl.show()

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
    retval=G_V*G_V*xsec_SM*1e-6*(1.97e-14)*(1.97e-14)*HelmFormFactor(E_R, A)
    if(retval<=0.0):
        print(E_nu,E_R,ERmax(E_nu,A_Ge))
    return retval




def differentialRate_CEvNS(E_R, A, Z, gsq=0.0, m_med=1000.0):
    integrand = lambda E_nu: xsec_CEvNS(E_R, E_nu, A, Z, gsq, m_med)\
                        *neutrino_flux_nakazato(E_nu)
    E_min = (E_R*1e-3+np.sqrt(2.0*A*0.9315*E_R+E_R*E_R*1e-6))/2.0
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

mass = 1.0 #target mass in kg
time = 1.0 #exposure time in days

dist = 1.0
Pth=1.0
#norm=(Pth/3950.)*(30.0**2)/(dist**2)
norm=1.0
# Normalised for the flux https://doi.org/10.1007/JHEP04(2020)054 CONNIE 
# Count rates reproduced for Miner NIMA 853 (2017) 53

#COHERENT EVENT RATE vs Recoil Energy 
weight=1.0


####Checking the cross section vs neutrino energy
##enu=np.linspace(5,55,200)
##er=np.linspace(0,500,1000)
##anu=np.zeros(len(enu))
##for i in range(len(enu)):
##    xsec=lambda x:xsec_CEvNS(x,enu[i],A_Ge,Z_Ge)
##    aer=np.zeros(len(er))
##    for j in range(len(er)-1):
##        if(er[j+1]<=ERmax(enu[i],A_Ge)):
##            aer[j]=quad(xsec,er[j],er[j+1])[0]
##    anu[i]=np.sum(aer)
##pl.plot(enu,anu*1.0e38,enu,anu*1.0e38*2/3.0)
##pl.yscale('log')
##pl.xlim(5,55)
##pl.ylim(3.0e-6,400)
##pl.show()

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
pl.semilogy(E_R1, p2, label=r"Ge")
pl.semilogy(E_R1,p3,label=r"PbWO4")
#pl.loglog(E_R1,p4,label=r"Pb")
#pl.loglog(E_R1,p5,label=r"W")
#pl.loglog(E_R1,p6,label=r"O")



pl.ylim(1e-2, 1e2)
pl.xlim(0.01,50)
pl.xlabel("Recoil Energy (keV)")
pl.ylabel("[events/kg/keV]")
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

##pl.semilogy(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=3.0e-10)/1e6, label=r"$\mu_{\nu} / \mu_B = 3.0 X 10^{-10}$")
##pl.semilogy(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=1.0e-10)/1e6, label=r"$\mu_{\nu} / \mu_B = 1.0 X 10^{-10}$")
##pl.semilogy(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=5.0e-11)/1e6, label=r"$\mu_{\nu} / \mu_B = 5.0 X 10^{-11}$")
##pl.semilogy(E_R, mass*time*diffRate_full(E_R, A_Ge, Z_Ge, mu_nu=1.0e-12)/1e6, label=r"$\mu_{\nu} / \mu_B = 1.0 X 10^{-12}$")
##pl.axvline(0.1, linestyle='--', color='k')
##pl.ylim(1e-9, 1e0)
##pl.xlim(1e-2, 50)
##pl.xlabel("Recoil Energy (keV)")
##pl.ylabel("[events/10kg/keV/yr] x $10^6$")
##pl.legend( fontsize=12)
##pl.savefig("COHERENT_magnetic-mom.pdf", bbox_inches="tight")
##pl.show()


dRdPE_Si = lambda x: mass*norm*time*diffRate_CEvNS(x, A_Si, Z_Si)
dRdPE_Ge = lambda x: mass*norm*time*diffRate_CEvNS(x, A_Ge, Z_Ge)
#dRdPE_Ar = lambda x: mass*norm*time*diffRate_CEvNS(x, A_Ar, Z_Ar)
dRdPE_PbWO4 = lambda x: mass*norm*time*(f_Pb*diffRate_CEvNS(x, A_Pb, Z_Pb)+f_W*diffRate_CEvNS(x, A_W, Z_W)+f_O*diffRate_CEvNS(x, A_O, Z_O))
PE_bins = np.linspace(0.01,50,101)

N_SM_Ge = np.zeros(1000)
N_SM_Si = np.zeros(1000)
N_SM_PbWO4 = np.zeros(1000)
print(5)
for i in range(100):
    if(i%100==0):
        print(i)
    N_SM_Ge[i] = quad(dRdPE_Ge, PE_bins[i], PE_bins[i+1], epsabs = 0.01)[0]
    N_SM_Si[i] = quad(dRdPE_Si, PE_bins[i], PE_bins[i+1], epsabs = 0.01)[0]
    N_SM_PbWO4[i] = quad(dRdPE_PbWO4, PE_bins[i], PE_bins[i+1], epsabs = 0.01)[0]

##pl.step(PE_bins, np.append(N_SM_tot,0), 'g', linestyle="-", where = "post", label="CEvNS signal (this work)",linewidth=1.5)
##pl.axhline(0, linestyle='--', color = 'gray')
##pl.xlabel("Number of photoelectrons (PE)")
##pl.ylabel("Expected Counts")
##pl.legend( fontsize=14)
##pl.xlim(0, 10)
##pl.show()

print("Total CEvNS events expected (E_NR cutoff) Ge: ", np.sum(N_SM_Ge))
print("Total CEvNS events expected (E_NR cutoff) Si: ", np.sum(N_SM_Si))
print("Total CEvNS events expected (E_NR cutoff) PbWO4: ", np.sum(N_SM_PbWO4))

def YL(E_R,Z,k): #Lindhard Ionization yeild
    eps=11.5*(Z**(-7.0/3.0))*E_R
    g  =3.0*(eps**0.15)+0.7*(eps**0.6)+eps
    return k*g/(1.0+k*g)

def YGe(E_R,Z,k,eff): #Ionization Yeild for "eff" ionization efficiency
    if(eff=="Hi"):
        ll=0.015
        expdiv=0.07103
    elif(eff=="Fid"):
        ll=0.040
        expdiv=0.0609
    elif(eff=="Low"):
        ll=0.090
        expdiv=0.04242
        
    if(E_R<=ll):
        #print(E_R)
        return 0
    elif(E_R<254e-3 and E_R>ll):
        return 0.18*(1.0-np.exp(-(E_R-ll)/expdiv))
    elif(E_R>=254e-3):
        return YL(E_R,Z,k)
    
def dYL(E_R,Z,k): #derivative of YL
    eps=11.5*(Z**(-7.0/3.0))*E_R
    g  =3.0*(eps**0.15)+0.7*(eps**0.6)+eps
    t1 =k/(1.0+k*g)-(g*k*k/(1.0+k*g)**2.0)
    t2 =0.45*(eps**(-0.85))+0.42*(eps**(-0.4))+1.0
    t3=11.5*Z**(-7/3)
    return t1*t2*t3
    
def dYGedE_R(E_R,Z,k,eff):
    if(eff=="Hi"):
        ll=0.015
        expdiv=0.07103
    elif(eff=="Fid"):
        ll=0.040
        expdiv=0.0609
    elif(eff=="Low"):
        ll=0.090
        expdiv=0.04242
    
    if(E_R<=ll):
        return 0
    elif(E_R<254e-3 and E_R>ll):
        return 0.18*(np.exp(-(E_R-ll)/expdiv))/expdiv
    elif(E_R>=0.254):
        return dYL(E_R,Z,k)
    


def QSi(E_I):
    return (168.0*E_I+156.0*E_I**2+E_I**3)/(56.0+1097.0*E_I+383.0*E_I**2)
def dQSi(E_I):
    t1=(168.0+2.0*156.0*E_I+3.0*E_I**2)/(56.0+1097.0*E_I+383.0*E_I**2)
    t2=(168.0*E_I+156.0*E_I**2+E_I**3)*(1097.0+2.0*383.0*E_I)/((56.0+1097.0*E_I+383.0*E_I**2.0)**2.0)
    return t1-t2
def diffrentialRate_CEvNS_Si_fougel(x,A,Z):
    t1=1.0-x/QSi(x)*dQSi(x)
    enr=x/QSi(x)
    return differentialRate_CEvNS(enr, A, Z)*t1/QSi(x)

diffRate_CEvNS_Si=np.vectorize(diffrentialRate_CEvNS_Si_fougel)



eps = lambda x: (0.845-1.0/(1.0+np.exp(42.66*(x-0.067))))
dRdEE_Si = lambda x: mass*norm*time*diffRate_CEvNS_Si(x, A_Si, Z_Si)*eps(x)
print(7)
nbin=100
PE_bins_Q = np.logspace(np.log10(0.01),np.log10(30),nbin+1)
N_SM_Si_Q = np.zeros(nbin)
pl.plot(PE_bins_Q,diffRate_CEvNS(PE_bins_Q, 79, 54))
pl.xlim(1,30)
pl.show()



for i in range(nbin):
    N_SM_Si_Q[i] = quad(dRdEE_Si, PE_bins_Q[i], PE_bins_Q[i+1], epsabs = 0.01)[0]

EconvGedat=np.loadtxt("Ge_Hi_EI_y_vs_ENR_x.dat")
Eion_bins = np.arange(0.01,EconvGedat[-1,1],0.0029)
N_SM_Ge_Q = np.zeros(len(Eion_bins)-1)
EconvGeVal = InterpolatedUnivariateSpline(EconvGedat[:,1], EconvGedat[:,0], k = 3)
def diffrentialRate_CEvNS_ion(x, A, Z, k=0.2, eff="Hi", conv=EconvGeVal):
    ##t1=1/YGe(x, Z_Ge, k, eff)-x*dYGedE_R(x, Z_Ge, k, eff)/(YGe(x, Z_Ge, k, eff)**2)
    xnr=EconvGeVal(x)
    t1=YGe(xnr, Z, k, eff)+xnr*dYGedE_R(xnr, Z, k, eff)
    if(t1==0.0):
        return 0.0
    else:
        return differentialRate_CEvNS(xnr, A, Z)/t1
diffRate_CEvNS_ion=np.vectorize(diffrentialRate_CEvNS_ion)
dRdEE_Ge = lambda x: mass*norm*time*diffRate_CEvNS_ion(x, A_Ge, Z_Ge, 0.2, "Hi")
eminindx=np.argmin(np.abs(Eion_bins-EconvGedat[0,1])) #Since the data file does not have ENR below the lower limit thus finding the ion energy closest to the required value 
print(len(Eion_bins))
for i in range(eminindx,len(Eion_bins)-1):#
    if(i%1000==0):
        print(i,Eion_bins[i],EconvGeVal(Eion_bins[i]))
    N_SM_Ge_Q[i] = quad(dRdEE_Ge, Eion_bins[i],Eion_bins[i+1], epsabs = 0.01)[0]
    
print("Total CEvNS events expected (E_ion cutoff) Ge: ", np.sum(N_SM_Ge_Q))
print("Total CEvNS events expected (E_ion cutoff) Si: ", np.sum(N_SM_Si_Q))
##ne=(Eion_bins-0.00067)/0.0029
##print(N_SM_Ge_Q)
##pl.bar(ne[1:],N_SM_Ge_Q)
##pl.xscale('log')
##pl.yscale('log')
##pl.show()

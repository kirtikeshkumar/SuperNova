# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:17:24 2024

@author: Kirtikesh Kumar
"""

import numpy as np
import matplotlib.pyplot as pl

"""
##For plotting Luminosity and Flux vs Time

fname="intp2001"
file=np.loadtxt(fname+"_Flux_Luminosity.dat")

pl.semilogx(file[:,0],file[:,1],color="red",label=r"$\nu_e$",linewidth=2)
pl.semilogx(file[:,0],file[:,2],color="green",label=r"$\bar(\nu)_e$",linewidth=2)
pl.semilogx(file[:,0],file[:,3],color="blue",label=r"$\nu_x$",linewidth=2)
pl.xlabel("Time after bounce (in s)",fontsize=20)
pl.ylabel("Neutrino Flux "+r"($MeV^{-1}$)",fontsize=20)
pl.ylim(0,3E57)
pl.yticks(fontsize=20)
pl.xticks(fontsize=20)
pl.legend(fontsize=20)
pl.savefig(fname+"_Flux.pdf", bbox_inches="tight")
pl.show()

pl.semilogx(file[:,0],file[:,4],color="red",label=r"$\nu_e$",linewidth=2)
pl.semilogx(file[:,0],file[:,5],color="green",label=r"$\bar{\nu}_e$",linewidth=2)
pl.semilogx(file[:,0],file[:,6],color="blue",label=r"$\nu_x$",linewidth=2)
pl.xlabel("Time after bounce (in s)",fontsize=20)
pl.ylabel(r"$L_{\nu}$ "+r"($MeV^{-1}$)",fontsize=20)
pl.ylim(0,5E52)
pl.yticks(fontsize=20)
pl.xticks(fontsize=20)
pl.legend(fontsize=20)
pl.savefig(fname+"_Luminosity.pdf", bbox_inches="tight")
pl.show()
"""


mass="20"
bouncetime="100"
filename="SNn_TimeIntegrated_"+mass+"M"+bouncetime
data=np.loadtxt(filename+".dat")
bins=np.append(data[:,0],data[-1,1])
pl.hist(bins[:-1], bins, weights=data[:,2],histtype='step',label=r'$\nu_e$',color='red',linewidth=3)
pl.hist(bins[:-1], bins, weights=data[:,3],histtype='step',label=r'$\bar{\nu}_e$',color='green',linewidth=3)
pl.hist(bins[:-1], bins, weights=data[:,4],histtype='step',label=r'$\nu_x$',color='blue',linewidth=3)
pl.xlim(0,50)
pl.xlabel("neutrino energy [MeV]")
pl.ylabel("total no. of neutrino [MeV$^{-1}$]")
pl.legend( fontsize=14)
pl.savefig(filename+"_NuSpec.pdf", bbox_inches="tight")
pl.show()

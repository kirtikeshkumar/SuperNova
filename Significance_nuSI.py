import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator


plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 3,  # Thicker edges for the plot
    'axes.titlesize': 35,         # Font size for the plot title
    'axes.labelsize': 22,         # Font size for x and y labels
    'axes.labelpad': 12,          # spacing between axis label and axis
    'legend.fontsize': 24,        # Font size for legend text
    'legend.title_fontsize': 26,  # Font size for legend title
    'xtick.direction': 'in',  # Ticks pointing inside
    'ytick.direction': 'in',  # Ticks pointing inside
    'xtick.major.size': 10,      # Length of major ticks
    'ytick.major.size': 10,      # Length of major ticks
    'xtick.major.width': 2,      # Thickness of major ticks
    'ytick.major.width': 2,      # Thickness of major ticks
    'xtick.minor.size': 5,       # Length of minor ticks
    'ytick.minor.size': 5,       # Length of minor ticks
    'xtick.minor.width': 1,      # Thickness of minor ticks
    'ytick.minor.width': 1,      # Thickness of minor ticks
    'xtick.labelsize': 17,       # Font size for tick labels
    'ytick.labelsize': 17,       # Font size for tick labels
})

massInCode  = 10.0      ## the mass of detector for which counts are calculated
distInCode  = 196       ## the distance of source for which counts are calculated

dist        = np.logspace(-1,3,1000)/1000 #dist in kpc
mass        = np.linspace(0.001,100,1000)#np.array([10.0])#
X,Y         = np.meshgrid(dist,mass)
mu          = np.array([j/i**2/1000**2 for j in mass for i in dist]) #factor of 1000**2 for distance conversion from kpc to pc

norm        = 1.0/massInCode * distInCode**2 * mu

bgnorm      = 500 * 20/86400 * mass #500 from 0.5 MeV energy range, 20/86400 from 20 seconds
bgnormext   = np.array([])
for i in range(len(dist)):
    bgnormext = np.append(bgnormext,bgnorm)
bgnormext   = bgnormext.reshape(len(dist),len(mass))
bgnormext   = bgnormext.transpose()
bgnorm      = bgnormext.reshape(len(mass)*len(dist))
bgrates      = np.array([10])

## Standard Counts
Ge30  = 434
Ge100 = 407
Sap50 = 283

## Number Conserving
Ge30NCon  = 482
Ge100NCon = 463
Sap50NCon = 320

## Number Changing
Ge30NCh  = 462
Ge100NCh = 441
Sap50NCh = 305

alpha = 1.0 # ratio of time for measuring On and Off signals
alpha2 = 1.0 + alpha
alpha1 = alpha2/alpha

for bgr in bgrates:
    bgrate = bgr * bgnorm
    sigGe30  = Ge30  * norm + bgrate
    sigGe100 = Ge100 * norm + bgrate
    sigSap50 = Sap50 * norm + bgrate

    sigGe30NCon  = Ge30NCon  * norm + bgrate
    sigGe100NCon = Ge100NCon * norm + bgrate
    sigSap50NCon = Sap50NCon * norm + bgrate

    sigGe30NCh  = Ge30NCh  * norm + bgrate
    sigGe100NCh = Ge100NCh * norm + bgrate
    sigSap50NCh = Sap50NCh * norm + bgrate

    ZGe30wNCon  = np.sqrt(2*(sigGe30NCon*np.log(alpha1*sigGe30NCon/(sigGe30+sigGe30NCon)) + sigGe30*np.log(alpha2*sigGe30/(sigGe30+sigGe30NCon))))
    ZGe100wNCon = np.sqrt(2*(sigGe100NCon*np.log(alpha1*sigGe100NCon/(sigGe100+sigGe100NCon)) + sigGe100*np.log(alpha2*sigGe100/(sigGe100+sigGe100NCon))))
    ZSap50wNCon = np.sqrt(2*(sigSap50NCon*np.log(alpha1*sigSap50NCon/(sigSap50+sigSap50NCon)) + sigSap50*np.log(alpha2*sigSap50/(sigSap50+sigSap50NCon))))

    ZGe30NChwNCon  = np.sqrt(2*(sigGe30NCon*np.log(alpha1*sigGe30NCon/(sigGe30NCh+sigGe30NCon)) + sigGe30NCh*np.log(alpha2*sigGe30NCh/(sigGe30NCh+sigGe30NCon))))
    ZGe100NChwNCon = np.sqrt(2*(sigGe100NCon*np.log(alpha1*sigGe100NCon/(sigGe100NCh+sigGe100NCon)) + sigGe100NCh*np.log(alpha2*sigGe100NCh/(sigGe100NCh+sigGe100NCon))))
    ZSap50NChwNCon = np.sqrt(2*(sigSap50NCon*np.log(alpha1*sigSap50NCon/(sigSap50NCh+sigSap50NCon)) + sigSap50NCh*np.log(alpha2*sigSap50NCh/(sigSap50NCh+sigSap50NCon))))

    ZS50    = ZSap50wNCon.reshape(len(mass),len(dist))
    ZGe30   = ZGe30wNCon.reshape(len(mass),len(dist))
    ZGe100  = ZGe100wNCon.reshape(len(mass),len(dist))

    ZS50SI    = ZSap50NChwNCon.reshape(len(mass),len(dist))
    ZGe30SI   = ZGe30NChwNCon.reshape(len(mass),len(dist))
    ZGe100SI  = ZGe100NChwNCon.reshape(len(mass),len(dist))
    

    fig,ax=plt.subplots(2,1, figsize=(6,12), constrained_layout=True)
    levels = np.append(1,np.linspace(5.000, 50.0, num=10))
    levels = np.append(levels, 500)

    a1=ax[1].contourf(X,Y,ZGe100SI,levels=levels,vmin=0,vmax=50,cmap = "cool")
    ax[1].contour(X, Y, ZGe100SI, levels=[5], colors='blue', linewidths=4)
    ax[1].contour(X, Y, ZGe100SI, levels=[3], colors='black', linestyles='dashed', linewidths=4)
    ax[1].set_xlim(0.001,0.2)
    ax[1].grid(alpha=0.5)
    ax[1].text(0.17, 4, "(b)", fontsize=25, fontweight='bold')

    a0=ax[0].contourf(X,Y,ZGe100,levels=levels,vmin=0,vmax=50,cmap = "cool")
    ax[0].contour(X, Y, ZGe100, levels=[5], colors='blue', linewidths=4)
    ax[0].contour(X, Y, ZGe100, levels=[3], colors='black', linestyles='dashed', linewidths=4)
    ax[0].set_xlim(0.001,0.5)
    ax[0].grid(alpha=0.5)
    ax[0].text(0.425, 4, "(a)", fontsize=25, fontweight='bold')

    fig.supylabel("detector mass (kg)",fontsize=22, fontweight='bold')
    fig.supxlabel("distance (kpc)",fontsize=22, fontweight='bold')
    plt.show()
    
##    fig,ax1=plt.subplots(figsize=(9,7.1))
##    a1 = ax1.contourf(X,Y,ZGe100,levels=levels,vmin=0,vmax=50,cmap = "cool")
##    ax1.contour(X, Y, ZGe100, levels=[5], colors='blue', linewidths=4)
##    ax1.contour(X, Y, ZGe100, levels=[3], colors='black', linestyles='dashed', linewidths=4)
##    fig.colorbar(a1)
##    ax1.set_xlabel("distance (kpc)")
##    ax1.set_ylabel("detector mass (kg)")
##    ax1.grid(alpha=0.5)
##    ax1.set_xlim(0.001,0.5)
##    plt.show()
##
##    fig,ax1=plt.subplots(figsize=(9,7.1))
##    a1 = ax1.contourf(X,Y,ZGe100SI,levels=levels,vmin=0,vmax=50,cmap = "cool")
##    ax1.contour(X, Y, ZGe100SI, levels=[5], colors='blue', linewidths=4)
##    ax1.contour(X, Y, ZGe100SI, levels=[3], colors='black', linestyles='dashed', linewidths=4)
##    fig.colorbar(a1)
##    ax1.set_xlabel("distance (kpc)")
##    ax1.set_ylabel("detector mass (kg)")
##    ax1.grid(alpha=0.5)
##    ax1.set_xlim(0.001,0.2)
##    plt.show()
    

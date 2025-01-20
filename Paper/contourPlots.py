import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 3,  # Thicker edges for the plot
    'axes.titlesize': 35,         # Font size for the plot title
    'axes.labelsize': 30,         # Font size for x and y labels
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
    'xtick.labelsize': 24,       # Font size for tick labels
    'ytick.labelsize': 24,       # Font size for tick labels
})


dist=np.logspace(2,3.31,1000)
mass=np.linspace(0.001,100,1000)
detmass = 10.0
countsSapphire=np.array([115/detmass*j*196*196/i/i for j in mass for i in dist])
countsGe100eV=np.array([171.06/detmass*j*196*196/i/i for j in mass for i in dist])
countsGe10eV=np.array([424.94/detmass*j*196*196/i/i for j in mass for i in dist])

X,Y=np.meshgrid(dist,mass)
ZSapphire=countsSapphire.reshape(1000,1000)
ZGe100eV=countsGe100eV.reshape(1000,1000)
ZGe10eV=countsGe10eV.reshape(1000,1000)

#plt.pcolor(X,Y,ZSapphire)

mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15) 
fig,ax=plt.subplots(1,3,sharey=True)
a0=ax[0].contourf(X,Y,np.log10(ZSapphire),levels=30,vmin=0,vmax=5,cmap = "plasma")
a1=ax[1].contourf(X,Y,np.log10(ZGe10eV),levels=30,vmin=0,vmax=5,cmap = "plasma")
a2=ax[2].contourf(X,Y,np.log10(ZGe100eV),levels=30,vmin=0,vmax=5,cmap = "plasma")
ax[0].set_title(r"Sapphire $E_{th} = 50eV_t$",fontsize=20)
ax[0].set_ylabel("detector mass (kg)",fontsize=24)
#ax[1].set_ylabel("detector mass (kg)")
#ax[2].set_ylabel("detector mass (kg)")
ax[1].set_title("Ge , Lindhard, $E_{th} = 50eV_t$",fontsize=20)
ax[2].set_title("Ge , Nakazato, $E_{th} = 50eV_t$",fontsize=20)
ax[0].set_xlabel("supernova distance (pc)",fontsize=15)
ax[1].set_xlabel("supernova distance (pc)",fontsize=15)
ax[2].set_xlabel("supernova distance (pc)",fontsize=15)
fig.colorbar(a2,ax=ax[2])
#plt.colorbar()
#plt.xlabel("distance (pc)")
#plt.ylabel("detector mass (kg)")
plt.show()

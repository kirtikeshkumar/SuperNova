import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator


plt.rcParams.update({
    'font.weight': 'bold',
    'font.family': 'serif',  
    'font.serif': ['Times New Roman'],  # Specify Times New Roman
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

dist        = np.logspace(-1,3.48,1000)/1000 #dist in kpc
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
bgrates      = np.array([10, 100])

ZS100 = {}
ZG10 = {}

alpha = 20.0/86400 # ratio of time for measuring On and Off signals
alpha2 = 1.0 + alpha
alpha1 = alpha2/alpha


for bgr in bgrates:
    bgrate = bgr * bgnorm
    sigSap50    = 115 * norm + bgrate
    sigSap100   = 228 * norm + bgrate
    
    sigGe10ee   = 177 * norm + bgrate
    sigGe100ee  = 435 * norm + bgrate
    sigGe170ee  = 158 * norm + bgrate
    bgrate = bgrate/alpha
    
    ## using Li Ma statistic from https://www.mpe.mpg.de/~ste/data/aa0839.pdf
    
    Z0muSap50   = np.sqrt(2*(sigSap50*np.log(alpha1*sigSap50/(bgrate+sigSap50)) + bgrate*np.log(alpha2*bgrate/(bgrate+sigSap50))))##np.sqrt(2*(sigSap50*np.log(sigSap50/bgrate) + bgrate - sigSap50)) ##these commented ones need to be rederived in fair.
    Z0muSap100  = np.sqrt(2*(sigSap100*np.log(alpha1*sigSap100/(bgrate+sigSap100)) + bgrate*np.log(alpha2*bgrate/(bgrate+sigSap100))))##np.sqrt(2*(sigSap100*np.log(sigSap100/bgrate) + bgrate - sigSap100))
    
    Z0muGe10    = np.sqrt(2*(sigGe10ee*np.log(alpha1*sigGe10ee/(bgrate+sigGe10ee)) + bgrate*np.log(alpha2*bgrate/(bgrate+sigGe10ee)))) ##np.sqrt(2*(sigGe10ee*np.log(sigGe10ee/bgrate) + bgrate - sigGe10ee))
    Z0muGe100   = np.sqrt(2*(sigGe100ee*np.log(alpha1*sigGe100ee/(bgrate+sigGe100ee)) + bgrate*np.log(alpha2*bgrate/(bgrate+sigGe100ee))))
    Z0muGe170   = np.sqrt(2*(sigGe170ee*np.log(alpha1*sigGe170ee/(bgrate+sigGe170ee)) + bgrate*np.log(alpha2*bgrate/(bgrate+sigGe170ee))))##np.sqrt(2*(sigGe170ee*np.log(sigGe170ee/bgrate) + bgrate - sigGe170ee))

    
    ZSapphire50 = Z0muSap50.reshape(len(mass),len(dist))
    ZSapphire100= Z0muSap100.reshape(len(mass),len(dist))
    
    ZGe10       = Z0muGe10.reshape(len(mass),len(dist))
    ZGe100      = Z0muGe100.reshape(len(mass),len(dist))
    ZGe170      = Z0muGe170.reshape(len(mass),len(dist))
    
    ZS100[bgr] = ZSapphire100.tolist()
    ZG10[bgr] = ZGe10.tolist()
    
##    mpl.rc('xtick', labelsize=15)
##    mpl.rc('ytick', labelsize=15) 
    fig,ax=plt.subplots(2,1,sharex=True, figsize=(6,12), constrained_layout=True)
##    ##levels = np.logspace(np.log10(1.0), np.log10(100), num=10)
    levels = np.append(1,np.linspace(5.000, 50.0, num=10))
    levels = np.append(levels, 500)
    ##levels=np.array([0.0,0.1,0.5,0.7,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,10.0,25.0,50.0,1000.0])
    ##a0=ax[0].contourf(X,Y,np.log10(ZSapphire100),levels=levels,vmin=0,vmax=500,cmap = "plasma")
    ##a1=ax[1].contourf(X,Y,np.log10(ZGe10),levels=levels,vmin=0,vmax=500,cmap = "plasma")
    ##a2=ax[2].contourf(X,Y,np.log10(ZGe100),levels=levels,vmin=0,vmax=500,cmap = "plasma")
    a2=ax[1].contourf(X,Y,ZSapphire50,levels=levels,vmin=0,vmax=50,cmap = "cool")
    a0=ax[0].contourf(X,Y,ZGe10,levels=levels,vmin=0,vmax=50,cmap = "cool")
    ##a1=ax[1].contourf(X,Y,ZGe100,levels=levels,vmin=0,vmax=50,cmap = "plasma")

    ax[1].contour(X, Y, ZSapphire50, levels=[5], colors='blue', linewidths=4)
    ax[0].contour(X, Y, ZGe10, levels=[5], colors='blue', linewidths=4)
    ##ax[1].contour(X, Y, ZGe100, levels=[5], colors='green', linestyles='dashed', linewidths=4)

    ax[1].contour(X, Y, ZSapphire50, levels=[3], colors='black', linestyles='dashed', linewidths=4)
    ax[0].contour(X, Y, ZGe10, levels=[3], colors='black', linestyles='dashed', linewidths=4)
    ##ax[1].contour(X, Y, ZGe100, levels=[3], colors='black', linewidths=4)

    fig.supylabel("detector mass (kg)",fontsize=22, fontweight='bold', fontname="Times New Roman")

##    ax[1].set_title(r"$Sapphire, E_{th} = 50eV_t$",fontsize=30)
##    ax[0].set_ylabel("detector mass (kg)")#,fontsize=24)
    #ax[1].set_ylabel("detector mass (kg)")
    #ax[2].set_ylabel("detector mass (kg)")
##    ax[0].set_title("$Ge, E_{th} = 30eV_{ee}$",fontsize=30)
    ##ax[1].set_title("$Ge , Livermore, E_{th} = 50eV_{ee}$",fontsize=20)
    ax[1].set_xlabel("distance (kpc)")#,fontsize=15)
##    ax[0].set_xlabel("distance (kpc)")#,fontsize=15)
    ##ax[1].set_xlabel("distance (kpc)")#,fontsize=15)
    ax[1].set_xlim(0.1,1.5)
    ax[0].set_xlim(0.1,1.5)
    ##ax[1].set_xlim(0.1,1.5)
    fig.colorbar(a2,ax=ax[1])
    fig.colorbar(a0,ax=ax[0])
    ##plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    #plt.colorbar()
    #plt.xlabel("distance (pc)")
    #plt.ylabel("detector mass (kg)")
    ax[0].grid(alpha=0.5)
    ax[1].grid(alpha=0.5)
    ax[0].text(1.2, 4, "(a)", fontsize=25, fontweight='bold', fontname="Times New Roman")
    ax[1].text(1.2, 4, "(b)", fontsize=25, fontweight='bold', fontname="Times New Roman")

    plt.show()

##    fig,ax=plt.subplots(figsize=(9,7.1))
##    a0=ax.contourf(X,Y,ZGe10,levels=levels,vmin=0,vmax=50,cmap = "cool")
##    ax.contour(X, Y, ZGe10, levels=[5], colors='blue', linewidths=4)
##    ax.contour(X, Y, ZGe10, levels=[3], colors='black', linestyles='dashed', linewidths=4)
##    ax.set_ylabel("detector mass (kg)")
##    ax.set_xlabel("distance (kpc)")
##    ax.set_xlim(0.1,1.5)
##    ax.grid(alpha=0.5)
##    fig.colorbar(a0)
##    plt.show()
##
##    fig,ax=plt.subplots(figsize=(9,7.1))
##    a0=ax.contourf(X,Y,ZSapphire50,levels=levels,vmin=0,vmax=50,cmap = "cool")
##    ax.contour(X, Y, ZSapphire50, levels=[5], colors='blue', linewidths=4)
##    ax.contour(X, Y, ZSapphire50, levels=[3], colors='black', linestyles='dashed', linewidths=4)
##    ax.set_ylabel("detector mass (kg)")
##    ax.set_xlabel("distance (kpc)")
##    ax.set_xlim(0.1,1.5)
##    ax.grid(alpha=0.5)
##    fig.colorbar(a0)
##    plt.show()
    
# plt.plot(dist,ZGe10[0],label=f'{mass[0]} kg')
# plt.plot(dist,ZGe10[1],label=f'{mass[1]} kg')
# plt.legend()
# plt.ylabel("Significance of measurement")
# plt.xlabel("Distance of Supernova (kpc)")
# plt.yscale('log')
# plt.ylim(1,500)
# plt.title(f'background rate {round(bgrate[0]/bgnorm[0],1)} counts/keV/kg/day')
# plt.show()

plt.fill_between(dist, ZG10[10][0], ZG10[100][0], color='orange', alpha=0.3, label='10eV Ge')
plt.fill_between(dist, ZS100[10][0], ZS100[100][0], color='blue', alpha=0.3, label='100eV Sapphire')

plt.ylabel("Measurement Significance", fontsize=14)
plt.xlabel("Supernova Distance (kpc)", fontsize=14)
plt.yscale('log')
plt.ylim(1,100)
plt.xlim(0.1,1.5)
plt.hlines(5, 0, 1.5, color='black', linestyles='dashed')
plt.text(1.4, 5.5, s=r'5$\sigma$', color='black')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
##plt.savefig('SignificanceBands.pdf')
plt.show()

##countsSapphire=np.array([686.74/30*j*196*196/i/i for j in mass for i in dist])
##countsGe10eVNTL=np.array([631.14/30*j*196*196/i/i for j in mass for i in dist])
##countsGe10eV=np.array([366.8/10*j*196*196/i/i for j in mass for i in dist])
##
##ZSapphire=countsSapphire.reshape(1000,1000)
##ZGe10eVNTL=countsGe10eVNTL.reshape(1000,1000)
##ZGe10eV=countsGe10eV.reshape(1000,1000)
##
###plt.pcolor(X,Y,ZSapphire)
##
##mpl.rc('xtick', labelsize=15)
##mpl.rc('ytick', labelsize=15) 
##fig,ax=plt.subplots(1,3,sharey=True)
##a0=ax[0].contourf(X,Y,np.log10(ZSapphire),levels=30,vmin=-1,vmax=5,cmap = "plasma")
##a1=ax[1].contourf(X,Y,np.log10(ZGe10eV),levels=30,vmin=-1,vmax=5,cmap = "plasma")
##a2=ax[2].contourf(X,Y,np.log10(ZGe10eVNTL),levels=30,vmin=-1,vmax=5,cmap = "plasma")
##ax[0].set_title("Sapphire 100eVnr threshold",fontsize=20)
##ax[0].set_ylabel("detector mass (kg)",fontsize=24)
###ax[1].set_ylabel("detector mass (kg)")
###ax[2].set_ylabel("detector mass (kg)")
##ax[1].set_title("Ge 10eVnr threshold ",fontsize=20)
##ax[2].set_title("Ge 10eVee threshold",fontsize=20)
##ax[0].set_xlabel("supernova distance (pc)",fontsize=15)
##ax[1].set_xlabel("supernova distance (pc)",fontsize=15)
##ax[2].set_xlabel("supernova distance (pc)",fontsize=15)
##fig.colorbar(a2,ax=ax[2])
###plt.colorbar()
###plt.xlabel("distance (pc)")
###plt.ylabel("detector mass (kg)")
##plt.show()

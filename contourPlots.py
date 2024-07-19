import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator

massInCode  = 10.0      ## the mass of detector for which counts are calculated
distInCode  = 196       ## the distance of source for which counts are calculated

dist        = np.logspace(2,3.48,1000)/1000 #dist in kpc
mass        = np.array([10.0])#np.linspace(0.001,100,1000)#
X,Y         = np.meshgrid(dist,mass)
mu          = np.array([j/i**2/1000**2 for j in mass for i in dist]) #factor of 1000**2 for distance conversion from kpc to pc

norm        = 1.0/massInCode * distInCode**2 * mu

bgnorm      = 2000 * 20/86400 * mass #2000 from 2 MeV energy range, 20/86400 from 20 seconds
bgnormext   = np.array([])
for i in range(len(dist)):
    bgnormext = np.append(bgnormext,bgnorm)
bgnormext   = bgnormext.reshape(len(dist),len(mass))
bgnormext   = bgnormext.transpose()
bgnorm      = bgnormext.reshape(len(mass)*len(dist))
bgrates      = np.array([10, 100])

ZS100 = {}
ZG10 = {}

for bgr in bgrates:
    bgrate = bgr * bgnorm
    sigSap50    = 229 * norm + bgrate
    sigSap100   = 228 * norm + bgrate
    
    sigGe10ee   = 369 * norm + bgrate
    sigGe100ee  = 359 * norm + bgrate
    sigGe170ee  = 351 * norm + bgrate
    
    
    
    Z0muSap50   = np.sqrt(2*(sigSap50*np.log(sigSap50/bgrate) + bgrate - sigSap50))
    Z0muSap100  = np.sqrt(2*(sigSap100*np.log(sigSap100/bgrate) + bgrate - sigSap100))
    
    Z0muGe10    = np.sqrt(2*(sigGe10ee*np.log(sigGe10ee/bgrate) + bgrate - sigGe10ee))
    Z0muGe100   = np.sqrt(2*(sigGe100ee*np.log(sigGe100ee/bgrate) + bgrate - sigGe100ee))
    Z0muGe170   = np.sqrt(2*(sigGe170ee*np.log(sigGe170ee/bgrate) + bgrate - sigGe170ee))
    
    ZSapphire50 = Z0muSap50.reshape(len(mass),len(dist))
    ZSapphire100= Z0muSap100.reshape(len(mass),len(dist))
    
    ZGe10       = Z0muGe10.reshape(len(mass),len(dist))
    ZGe100      = Z0muGe100.reshape(len(mass),len(dist))
    ZGe170      = Z0muGe170.reshape(len(mass),len(dist))
    
    ZS100[bgr] = ZSapphire100.tolist()
    ZG10[bgr] = ZGe10.tolist()
    
    # mpl.rc('xtick', labelsize=15)
    # mpl.rc('ytick', labelsize=15) 
    # fig,ax=plt.subplots(1,3,sharey=True)
    # ##levels = np.logspace(np.log10(1.0), np.log10(100), num=10)
    # levels = np.append(np.linspace(1.000, 61, num=10),100)
    # ##levels=np.array([0.0,0.1,0.5,0.7,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,5.0,10.0,25.0,50.0,1000.0])
    # ##a0=ax[0].contourf(X,Y,np.log10(ZSapphire100),levels=levels,vmin=0,vmax=500,cmap = "plasma")
    # ##a1=ax[1].contourf(X,Y,np.log10(ZGe10),levels=levels,vmin=0,vmax=500,cmap = "plasma")
    # ##a2=ax[2].contourf(X,Y,np.log10(ZGe100),levels=levels,vmin=0,vmax=500,cmap = "plasma")
    # a0=ax[0].contourf(X,Y,ZSapphire100,levels=levels,vmin=0,vmax=50,cmap = "plasma")
    # a1=ax[1].contourf(X,Y,ZGe10,levels=levels,vmin=0,vmax=50,cmap = "plasma")
    # a2=ax[2].contourf(X,Y,ZGe100,levels=levels,vmin=0,vmax=50,cmap = "plasma")
    # ax[0].set_title("Sapphire 100eVnr threshold",fontsize=20)
    # ax[0].set_ylabel("detector mass (kg)",fontsize=24)
    # #ax[1].set_ylabel("detector mass (kg)")
    # #ax[2].set_ylabel("detector mass (kg)")
    # ax[1].set_title("Ge 10eVee threshold ",fontsize=20)
    # ax[2].set_title("Ge 100eVee threshold",fontsize=20)
    # ax[0].set_xlabel("supernova distance (pc)",fontsize=15)
    # ax[1].set_xlabel("supernova distance (pc)",fontsize=15)
    # ax[2].set_xlabel("supernova distance (pc)",fontsize=15)
    # ax[0].set_xlim(0.1,1.5)
    # ax[1].set_xlim(0.1,1.5)
    # ax[2].set_xlim(0.1,1.5)
    # fig.colorbar(a2,ax=ax[2])
    # ##plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    # #plt.colorbar()
    # #plt.xlabel("distance (pc)")
    # #plt.ylabel("detector mass (kg)")
    # plt.show()

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
plt.savefig('SignificanceBands.pdf')
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

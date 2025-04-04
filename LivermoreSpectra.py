import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d, UnivariateSpline,InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from scipy.optimize import minimize
from matplotlib.ticker import LogLocator


plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 3,  # Thicker edges for the plot
    'axes.titlesize': 35,         # Font size for the plot title
    'axes.labelsize': 37,         # Font size for x and y labels
    'axes.labelpad': 15,          # spacing between axis label and axis
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
    'xtick.labelsize': 28,       # Font size for tick labels
    'ytick.labelsize': 28,       # Font size for tick labels
})


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


##plt.plot(T_nu, fnue_L(T_nu))
##plt.plot(T_nu, fanue_L(T_nu))
##plt.plot(T_nu, fnux_L(T_nu))
##plt.show()
##
##plt.semilogy(T_nu, fnue_L(T_nu))
##plt.semilogy(T_nu, fanue_L(T_nu))
##plt.semilogy(T_nu, fnux_L(T_nu))
##plt.show()
##
##plt.loglog(T_nu, fnue_L(T_nu))
##plt.loglog(T_nu, fanue_L(T_nu))
##plt.loglog(T_nu, fnux_L(T_nu))
##plt.show()

########################################################################
##                   Evaluating Time Integrated Flux                  ##
########################################################################
intFlux_nue = []
intFlux_anue = []
intFlux_nux = []
intFlux = []

normalisation=1.0#/(45.96e41)

for E in E_nu:
    f_intFlux_nue = lambda t:diffflux(E,t,fnue_E,fnue_L)
    f_intFlux_anue = lambda t:diffflux(E,t,fanue_E,fanue_L)
    f_intFlux_nux = lambda t:diffflux(E,t,fnux_E,fnux_L)

    intFlux_nue.append(quad(f_intFlux_nue,T_nu[0],T_nu[-1],epsrel=1e-4)[0]*normalisation)
    intFlux_anue.append(quad(f_intFlux_anue,T_nu[0],T_nu[-1],epsrel=1e-4)[0]*normalisation)
    intFlux_nux.append(quad(f_intFlux_nux,T_nu[0],T_nu[-1],epsrel=1e-4)[0]*normalisation)

##normalisation = 1.0
data = np.loadtxt("data/supernova-spectrum_M_20_Z_2_rev_100.txt") #MeV vs No./MeV
bins=np.append(data[:,0],data[-1,1])


def fluxbb(E,E_mean):
    T=E_mean/3.15
    prefac = 3.35e55
    return E**2/(np.exp(E/T)+1)*prefac

fluxbb=np.vectorize(fluxbb)

##plt.xlim(0,50)
##plt.show()


fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
##pbb = ax.plot(E_nu,fluxbb(E_nu,11.1),label=r'$bb$',color='red',linewidth=4)
neutrino_flux_list = InterpolatedUnivariateSpline(np.sqrt(data[:,0]*data[:,1]), (data[:,2]+data[:,3]+data[:,4]*4)*normalisation, k = 1) #MeV vs 1/MeV/m^2
pNe = ax.hist(bins[:-1], bins, weights=data[:,2]*normalisation,histtype='step',ls='--',label=r'$\nu_e$',color='blue',linewidth=4)
pNbe = ax.hist(bins[:-1], bins, weights=data[:,3]*normalisation,histtype='step',ls='--',label=r'$\bar{\nu}_e$',color='green',linewidth=4)
pNx = ax.hist(bins[:-1], bins, weights=data[:,4]*normalisation,histtype='step',ls='--',label=r'$\nu_x$',color='red',linewidth=4)
pLe, = ax.plot(E_nu,intFlux_nue,label=r'$\nu_e$',color='blue',linewidth=4)
pLbe, = ax.plot(E_nu,intFlux_anue,label=r'$\bar{\nu}_e$',color='green',linewidth=4)
pLx, = ax.plot(E_nu,intFlux_nux,label=r'$\nu_x$',color='red',linewidth=4)

ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)  # Major gridlines
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3)  # Minor gridlines (for log scale)

plt.xlim(0.001,50)
plt.xlabel("neutrino energy [MeV]")
plt.ylabel(r"No. of $\nu$'s [MeV$^{-1}$]")


inset_x, inset_y = 35, 1.8e56  # Adjusted position

# Define table cell positions
cell_x_start, cell_y_start = inset_x, inset_y
cell_x_offset, cell_y_offset = 5, 0.2e56  

line_styles = {
    (0,0): ("blue", "--"),   # nue, Nak
    (0,1): ("blue", "-"),  # nue, Liv
    (1,0): ("green", "--"),     # anue, Nak
    (1,1): ("green", "-"),    # anue, Liv
    (2,0): ("red", "--"),   # nux, Nak
    (2,1): ("red", "-")      # nux, Liv
}

row_heights = [cell_y_start - i * cell_y_offset for i in range(4)]

# Draw table grid
for i in range(4):
    ax.plot([cell_x_start, cell_x_start + 2 * cell_x_offset], 
            [row_heights[i]] * 2, color='black', linewidth=1)

for j in range(3):
    ax.plot([cell_x_start + j * cell_x_offset] * 2, 
            [row_heights[0], row_heights[-1]], color='black', linewidth=1)

# Labels for table columns & rows
ax.text(cell_x_start + cell_x_offset / 2, row_heights[0] + 0.05e56, "Nak", fontsize=22, ha='center', fontweight='bold')
ax.text(cell_x_start + 3 * cell_x_offset / 2, row_heights[0] + 0.05e56, "Liv", fontsize=22, ha='center', fontweight='bold')

# Move "Ge", "Sap", "Alt" slightly to the left
labels = [r'$\nu_e$', r'$\bar{\nu}_e$', r'$\nu_x$']
for i, label in enumerate(labels):
    ax.text(cell_x_start - 0.5, (row_heights[i] + row_heights[i+1]) / 2, label, 
            fontsize=22, va='center', fontweight='bold', ha='right')

# Draw lines inside table cells
for (row, col), (color, style) in line_styles.items():
    x_pos = cell_x_start + (col + 0.5) * cell_x_offset
    y_pos = (row_heights[row] + row_heights[row+1]) / 2  # Midpoint for each row

    ax.plot([x_pos - 1.7, x_pos + 1.7], [y_pos, y_pos], color=color, linestyle=style, linewidth=4)

plt.show()


for i in range(len(intFlux_nue)):
    intFlux.append(intFlux_nue[i]+intFlux_anue[i]+intFlux_nux[i]*4.0)
plt.hist(bins[:-1], bins, weights=(data[:,2]+data[:,3]+data[:,4]*4)*normalisation,histtype='step',ls='--',label=r'Nakazato',color='blue',linewidth=2)
plt.plot(E_nu,intFlux,label=r'Livermore',color='Orange',linewidth=2)
plt.show()





import numpy as np
import matplotlib.pyplot as plt

time = np.loadtxt("Garching_Output/garching_pinched_info_key.dat")


for i in range(0,300):
    f = np.loadtxt("Garching_Output/pinched_"+str(i)+".dat")
    plt.clf()
    plt.ylim(0,3e7)
    plt.plot(f[:,0],f[:,1], color='blue', label=r'$\nu_e$')
    plt.plot(f[:,0],f[:,4], color='red', label=r'$\bar{\nu}_e$')
    plt.plot(f[:,0],f[:,2], color='black', label=r'$\nu_x$')

    plt.title('Nuetrino Flux at: '+str(time[i,1]))
    plt.legend()
    plt.pause(0.05)
    
plt.show()

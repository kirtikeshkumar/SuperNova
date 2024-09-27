import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

###########################################################################
##                       Getting the time values                         ##
###########################################################################
##folder_path = "../intp2001/"  
##time_values = []
##
### List all files in the folder
##files = os.listdir(folder_path)
##
### Iterate through each file
##for file_name in files:
##    # Split the filename by underscore and dot to extract the time value
##    parts = file_name.split('_')
##    if len(parts) == 2:
##        time_value = parts[1].split('.d')[0]  # Remove the extension
####        print(time_value,parts[1].split('.d')[1])
##        time_values.append(float(time_value))
##time_values.sort()
##time_values=np.array(time_values)
####print(time_values)
##
##deltime_values = np.array([])
##for i in range(len(time_values)):
##    if(i==0):
##        deltime = float(time_values[i+1])-float(time_values[i])
##    elif(i==len(time_values)-1):
##        deltime = float(time_values[i])-float(time_values[i-1])
##    else:
##        deltime = 0.5*(float(time_values[i+1])-float(time_values[i-1]))
##    deltime_values=np.append(deltime_values,deltime)
##
##timebins = np.array([round(time_values[0]-deltime_values[0]*0.5,4)])
##timebins = np.append(timebins,0.5*(time_values[1:]+time_values[:-1]))
##timebins = np.append(timebins,round(time_values[-1]-deltime_values[-1]*0.5,4))

time_values = np.append(np.linspace(0.02,0.1,41),np.linspace(0.11,16,1590))
############################################################################
##                         Getting Data From Files                        ##
############################################################################
files = os.listdir("./Livermore3D/")

f = pd.read_csv("./Livermore3D/"+files[0],delimiter='	',header=None)
ErecBins = np.array(f.iloc[0][:-1])
ErecMid = 0.5*(ErecBins[1:]+ErecBins[:-1])
shp = f.shape

histEr_0_15 = np.zeros([len(time_values),shp[1]-2])
histEr_15_30 = np.zeros([len(time_values),shp[1]-2])
histEr_30_45 = np.zeros([len(time_values),shp[1]-2])
histEr_45_53 = np.zeros([len(time_values),shp[1]-2])

for file_name in files:
    parts = file_name.split('_')
    lowtime,hitime = parts[3][2:-1].split('s-')
    lowE,hiE = parts[-1][:-4].split('-')
    loTIndex = np.where(abs(time_values-float(lowtime))<=1e-4)[0][0]
    hiTIndex = np.where(abs(time_values-float(hitime))<=1e-4)[0][0]
    sstring = '_'+lowE+'_'+hiE
    histname = [var_name for var_name in globals() if sstring in var_name]             ## find the variable with sstring in its name
##    print(file_name,'\t',lowtime,hitime,'\t',lowE,hiE,'\t',loTIndex,hiTIndex,'\t',histname[0])
    print(file_name)
    
    f = pd.read_csv("./Livermore3D/"+file_name,delimiter='	',header=None)
    shp = f.shape
    f = np.array(f.iloc[1:shp[0],0:shp[1]-2])
    globals()[histname[0]][loTIndex:hiTIndex+1] = f

hist0_15_file = np.vstack((ErecMid, histEr_0_15))
hist15_30_file = np.vstack((ErecMid, histEr_15_30))
hist30_45_file = np.vstack((ErecMid, histEr_30_45))
hist45_53_file = np.vstack((ErecMid, histEr_45_53))

############################################################################
##                                Plotting                                ##
############################################################################
x_grid,y_grid = np.meshgrid(ErecMid,time_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x_grid, y_grid, np.log10(histEr_30_45+0.1), lw=0.5, rstride=8, cstride=8, alpha=0.3)
##surf = ax.plot_surface(x_grid, y_grid, np.log10(histEr_30_45+0.1), cmap='hot')
##fig.colorbar(surf)
##cont = ax.contourf(x_grid, y_grid, np.log10(histEr_30_45+0.1), extend3d=True, cmap='viridis')
##scatter = ax.scatter(x_grid, y_grid, histEr_30_45)
##fig.colorbar(cont)
ax.set_xlabel('E_r (keV)')
ax.set_ylabel('Post Bounce Time (s)')
ax.set_zlabel('log10(Counts/10kg/s)')
##ax.set(zscale='log')

plt.title(r'$15 < E_\nu < 30 MeV$')
plt.show()

gridpoints_0_15 = []
gridpoints_15_30 = []
gridpoints_30_45 = []
gridpoints_45_53 = []
for x in range(len(time_values)):
    for y in range(len(ErecMid)):
        gridpoints_0_15.append([time_values[x],ErecMid[y],histEr_0_15[x,y]])
        gridpoints_15_30.append([time_values[x],ErecMid[y],histEr_15_30[x,y]])
        gridpoints_30_45.append([time_values[x],ErecMid[y],histEr_30_45[x,y]])
        gridpoints_45_53.append([time_values[x],ErecMid[y],histEr_45_53[x,y]])

np.savetxt('gridpoints_0_15_Livermore.txt', gridpoints_0_15, delimiter='\t', fmt='%.6e')
np.savetxt('gridpoints_15_30_Livermore.txt', gridpoints_15_30, delimiter='\t', fmt='%.6e')
np.savetxt('gridpoints_30_45_Livermore.txt', gridpoints_30_45, delimiter='\t', fmt='%.6e')
np.savetxt('gridpoints_45_53_Livermore.txt', gridpoints_45_53, delimiter='\t', fmt='%.6e')

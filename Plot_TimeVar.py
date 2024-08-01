import numpy as np
import matplotlib.pyplot as plt
import os


########################################################################
##                       Make list of all files                       ##
########################################################################
folder_path_Ge_Nak = "data/intp2001_Ge/Nakazato"
##folder_path_Sap = "data/intp3003_Sap/Nakazato"
folder_path_Ge_Liv = "data/intp2001_Ge/Livermore"
##folder_path_Sap = "data/intp3003_Sap/Livermore"
fnamesNak    = os.listdir(folder_path_Ge_Nak)
fnamesLiv    = os.listdir(folder_path_Ge_Liv)

def getInitTimeFromGeFileName(fname):
    split0 = fname.split("Ge")
    split1 = split0[1].split("s")
    return float(split1[0])

def getInitTimeFromSapFileName(fname):
    split0 = fname.split("Sap")
    split1 = split0[1].split("s")
    return float(split1[0])

fnamesNak.sort(key=getInitTimeFromGeFileName)
fnamesLiv.sort(key=getInitTimeFromGeFileName)

########################################################################
##                  Define variables to store values                  ##
########################################################################
timeGeLiv = []
deltGeLiv = []
timeGeNak = []
deltGeNak = []
timeSap = []
deltSap = []

meanGe10eVeeLiv  = []
meanGe100eVeeLiv = []
meanGe170eVeeLiv = []
meanGe10eVthLiv  = []
meanGe100eVthLiv = []
meanGe170eVthLiv = []

stdevGe10eVeeLiv  = []
stdevGe100eVeeLiv = []
stdevGe170eVeeLiv = []
stdevGe10eVthLiv  = []
stdevGe100eVthLiv = []
stdevGe170eVthLiv = []

meanGe10eVeeNak  = []
meanGe100eVeeNak = []
meanGe170eVeeNak = []
meanGe10eVthNak  = []
meanGe100eVthNak = []
meanGe170eVthNak = []

stdevGe10eVeeNak  = []
stdevGe100eVeeNak = []
stdevGe170eVeeNak = []
stdevGe10eVthNak  = []
stdevGe100eVthNak = []
stdevGe170eVthNak = []

meanSap50eVth  = []
meanSap100eVth = []

stdevSap50eVth  = []
stdevSap100eVth = []

########################################################################
##                        Read and store value                        ##
########################################################################
for fname in fnamesLiv:
    file = open(folder_path_Ge_Liv+"/"+fname)
    for line in file:
        lsplit = line.split("\t")
        try:
            dt = float(lsplit[1])
            timeGeLiv.append(float(lsplit[0]))
            deltGeLiv.append(dt)
            
            meanGe10eVeeLiv.append(float(lsplit[2])/100.0)
            meanGe100eVeeLiv.append(float(lsplit[3])/100.0)
            meanGe170eVeeLiv.append(float(lsplit[4])/100.0)
            
            stdevGe10eVeeLiv.append(float(lsplit[5])/100.0)
            stdevGe100eVeeLiv.append(float(lsplit[6])/100.0)
            stdevGe170eVeeLiv.append(float(lsplit[7])/100.0)
            
            meanGe10eVthLiv.append(float(lsplit[8])/100.0)
            meanGe100eVthLiv.append(float(lsplit[9])/100.0)
            meanGe170eVthLiv.append(float(lsplit[10])/100.0)
            
            stdevGe10eVthLiv.append(float(lsplit[11])/100.0)
            stdevGe100eVthLiv.append(float(lsplit[12])/100.0)
            stdevGe170eVthLiv.append(float(lsplit[13].split("\n")[0])/100.0)
        except:
##            print(lsplit)
            continue
    file.close()

for fname in fnamesNak:
    file = open(folder_path_Ge_Nak+"/"+fname)
    for line in file:
        lsplit = line.split("\t")
        try:
            dt = float(lsplit[1])
            timeGeNak.append(float(lsplit[0]))
            deltGeNak.append(dt)
            
            meanGe10eVeeNak.append(float(lsplit[2])/100.0)
            meanGe100eVeeNak.append(float(lsplit[3])/100.0)
            meanGe170eVeeNak.append(float(lsplit[4])/100.0)
            
            stdevGe10eVeeNak.append(float(lsplit[5])/100.0)
            stdevGe100eVeeNak.append(float(lsplit[6])/100.0)
            stdevGe170eVeeNak.append(float(lsplit[7])/100.0)
            
            meanGe10eVthNak.append(float(lsplit[8])/100.0)
            meanGe100eVthNak.append(float(lsplit[9])/100.0)
            meanGe170eVthNak.append(float(lsplit[10])/100.0)
            
            stdevGe10eVthNak.append(float(lsplit[11])/100.0)
            stdevGe100eVthNak.append(float(lsplit[12])/100.0)
            stdevGe170eVthNak.append(float(lsplit[13].split("\n")[0])/100.0)
        except:
##            print(lsplit)
            continue
    file.close()

##for fname in fnamesSap:
##    file = open(folder_path_Sap+"/"+fname)
##    for line in file:
##        lsplit = line.split("\t")
##        try:
##            dt = float(lsplit[1])
##            timeSap.append(float(lsplit[0]))
##            deltSap.append(dt)
##            
##            meanSap50eVth.append(float(lsplit[2]))
##            meanSap100eVth.append(float(lsplit[3]))
##            
##            stdevSap50eVth.append(float(lsplit[4]))
##            stdevSap100eVth.append(float(lsplit[5]))
##            
##        except:
####            print(lsplit)
####            print("reading from "+folder_path+"Ge/"+fname)
##            continue
##    file.close()


########################################################################
##                              Plotting                              ##
########################################################################
##plt.xscale("log")
plt.yscale("log")
plt.ylim(0.1,100)
##plt.errorbar(timeGeNak, meanGe10eVeeNak, yerr=stdevGe10eVeeNak, label=r'Nakazato')
##plt.errorbar(timeGeLiv, meanGe10eVeeLiv, yerr=stdevGe10eVeeLiv, label=r'Livermore')
plt.plot(timeGeNak, meanGe10eVeeNak, color='orange', label=r'Nakazato')
plt.plot(timeGeLiv, meanGe10eVeeLiv, color='blue', label=r'Livermore')
plt.fill_between(timeGeNak, np.array(meanGe10eVeeNak)+np.array(stdevGe10eVeeNak), np.array(meanGe10eVeeNak)-np.array(stdevGe10eVeeNak), label=r'Nakazato 1$\sigma$', color='orange', alpha=0.3)
plt.fill_between(timeGeLiv, np.array(meanGe10eVeeLiv)+np.array(stdevGe10eVeeLiv), np.array(meanGe10eVeeLiv)-np.array(stdevGe10eVeeLiv), label=r'Livermore 1$\sigma$', color='blue', alpha=0.3)
plt.xlabel("Post-bounce Time (in s)", fontsize=14)
plt.ylabel("Counts/kg/s", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
##plt.errorbar(timeGe, meanGe100eVee, yerr=stdevGe100eVee, label=r'threshold = 100eVee')
##plt.errorbar(timeGe, meanGe170eVee, yerr=stdevGe170eVee, label=r'threshold = 170eVee')
plt.legend()
plt.show()

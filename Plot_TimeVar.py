import numpy as np
import matplotlib.pyplot as plt
import os

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
########################################################################
##                       Make list of all files                       ##
########################################################################
folder_path_Ge_Nak = "data/intp2001_Ge/Nakazato_100kg"
folder_path_Sap_Nak = "data/intp2001_Sap/Nakazato_100kg"
folder_path_Ge_NakMixQF = "data/intp2003_Ge/Nakazato_100kg_MixQF"
folder_path_Ge_Liv = "data/intp2001_Ge/Livermore_100kg"
folder_path_Sap_Liv = "data/intp2003_Sap"#data/intp2001_Sap/Livermore_100kg"
fnamesNak    = os.listdir(folder_path_Ge_Nak)
fnamesNakMixQF    = os.listdir(folder_path_Ge_NakMixQF)
fnamesLiv    = os.listdir(folder_path_Ge_Liv)

##fnamesNak    = os.listdir(folder_path_Sap_Nak)
##fnamesLiv    = os.listdir(folder_path_Sap_Liv)


def getInitTimeFromGeFileName(fname):
    print(fname)
    split0 = fname.split("Ge")
    split1 = split0[1].split("s")
    return float(split1[0])

def getInitTimeFromGeMixQFFileName(fname):
    print(fname)
    split0 = fname.split("_")
    split4 = split0[4].split("s")
    return float(split4[0])

def getInitTimeFromSapFileName(fname):
    split0 = fname.split("Sap")
    split1 = split0[1].split("s")
    return float(split1[0])

fnamesNak.sort(key=getInitTimeFromGeFileName)
fnamesNakMixQF.sort(key=getInitTimeFromGeMixQFFileName)
fnamesLiv.sort(key=getInitTimeFromGeFileName)

##fnamesNak.sort(key=getInitTimeFromSapFileName)
##fnamesLiv.sort(key=getInitTimeFromSapFileName)


########################################################################
##                  Define variables to store values                  ##
########################################################################
timeGeLiv = []
deltGeLiv = []
timeGeNak = []
deltGeNak = []
timeGeNakMixQF = []
deltGeNakMixQF = []
timeSapNak = []
deltSapNak = []
timeSapLiv = []
deltSapLiv = []

meanGe10eVeeLiv  = []
meanGe100eVeeLiv = []
meanGe50eVeeLiv = []
meanGe10eVthLiv  = []
meanGe100eVthLiv = []
meanGe50eVthLiv = []
meanGe10eVtotLiv  = []
meanGe100eVtotLiv = []
meanGe50eVtotLiv = []

stdevGe10eVeeLiv  = []
stdevGe100eVeeLiv = []
stdevGe50eVeeLiv = []
stdevGe10eVthLiv  = []
stdevGe100eVthLiv = []
stdevGe50eVthLiv = []
stdevGe10eVtotLiv  = []
stdevGe100eVtotLiv = []
stdevGe50eVtotLiv = []

meanGe10eVeeNak  = []
meanGe100eVeeNak = []
meanGe50eVeeNak = []
meanGe10eVthNak  = []
meanGe100eVthNak = []
meanGe50eVthNak = []
meanGe10eVtotNak  = []
meanGe100eVtotNak = []
meanGe50eVtotNak = []

meanGe10eVeeNakMixQF  = []
meanGe100eVeeNakMixQF = []
meanGe50eVeeNakMixQF = []
meanGe10eVthNakMixQF  = []
meanGe100eVthNakMixQF = []
meanGe50eVthNakMixQF = []
meanGe10eVtotNakMixQF  = []
meanGe100eVtotNakMixQF = []
meanGe50eVtotNakMixQF = []

stdevGe10eVeeNakMixQF  = []
stdevGe100eVeeNakMixQF = []
stdevGe50eVeeNakMixQF = []
stdevGe10eVthNakMixQF  = []
stdevGe100eVthNakMixQF = []
stdevGe50eVthNakMixQF = []
stdevGe10eVtotNakMixQF  = []
stdevGe100eVtotNakMixQF = []
stdevGe50eVtotNakMixQF = []

stdevGe10eVeeNak  = []
stdevGe100eVeeNak = []
stdevGe50eVeeNak = []
stdevGe10eVthNak  = []
stdevGe100eVthNak = []
stdevGe50eVthNak = []
stdevGe10eVtotNak  = []
stdevGe100eVtotNak = []
stdevGe50eVtotNak = []

meanSap50eVthNak  = []
meanSap100eVthNak = []

stdevSap50eVthNak  = []
stdevSap100eVthNak = []

meanSap50eVthLiv  = []
meanSap100eVthLiv = []

stdevSap50eVthLiv  = []
stdevSap100eVthLiv = []

########################################################################
##                        Read and store value                        ##
########################################################################

########################################################################
##                             For Ge                                 ##
########################################################################

for fname in fnamesLiv:
    file = open(folder_path_Ge_Liv+"/"+fname)
    for line in file:
        lsplit = line.split("\t")
        try:
            dt = float(lsplit[1])
            timeGeLiv.append(float(lsplit[0]))
            deltGeLiv.append(dt)
            
            meanGe10eVeeLiv.append(float(lsplit[2])/10.0)
            meanGe100eVeeLiv.append(float(lsplit[3])/10.0)
            meanGe50eVeeLiv.append(float(lsplit[4])/10.0)
            
##            stdevGe10eVeeLiv.append(float(lsplit[5])/10.0)
##            stdevGe100eVeeLiv.append(float(lsplit[6])/10.0)
##            stdevGe50eVeeLiv.append(float(lsplit[7])/10.0)
            
            meanGe10eVthLiv.append(float(lsplit[8])/10.0)
            meanGe100eVthLiv.append(float(lsplit[9])/10.0)
            meanGe50eVthLiv.append(float(lsplit[10])/10.0)
            
##            stdevGe10eVthLiv.append(float(lsplit[11])/10.0)
##            stdevGe100eVthLiv.append(float(lsplit[12])/10.0)
##            stdevGe50eVthLiv.append(float(lsplit[13])/10.0)

            meanGe10eVtotLiv.append(float(lsplit[8])/10.0)
            meanGe100eVtotLiv.append(float(lsplit[9])/10.0)
            meanGe50eVtotLiv.append(float(lsplit[10])/10.0)
            
##            stdevGe10eVtotLiv.append(float(lsplit[11])/10.0)
##            stdevGe100eVtotLiv.append(float(lsplit[12])/10.0)
##            stdevGe50eVtotLiv.append(float(lsplit[13])/10.0)
        except:
##            print(lsplit)
            continue
    file.close()

for fname in fnamesNakMixQF:
    file = open(folder_path_Ge_NakMixQF+"/"+fname)
    for line in file:
        lsplit = line.split("\t")
        try:
            dt = float(lsplit[1])
            timeGeNakMixQF.append(float(lsplit[0]))
            deltGeNakMixQF.append(dt)
            
            meanGe10eVeeNakMixQF.append(float(lsplit[2])/10.0)
            meanGe100eVeeNakMixQF.append(float(lsplit[3])/10.0)
            meanGe50eVeeNakMixQF.append(float(lsplit[4])/10.0)
            
##            stdevGe10eVeeNakMixQF.append(float(lsplit[5])/10.0)
##            stdevGe100eVeeNakMixQF.append(float(lsplit[6])/10.0)
##            stdevGe50eVeeNakMixQF.append(float(lsplit[7])/10.0)
            
            meanGe10eVthNakMixQF.append(float(lsplit[8])/10.0)
            meanGe100eVthNakMixQF.append(float(lsplit[9])/10.0)
            meanGe50eVthNakMixQF.append(float(lsplit[10])/10.0)
            
##            stdevGe10eVthNakMixQF.append(float(lsplit[11])/10.0)
##            stdevGe100eVthNakMixQF.append(float(lsplit[12])/10.0)
##            stdevGe50eVthNakMixQF.append(float(lsplit[13])/10.0)   #.split("\n")[0]

            meanGe10eVtotNakMixQF.append(float(lsplit[8])/10.0)
            meanGe100eVtotNakMixQF.append(float(lsplit[9])/10.0)
            meanGe50eVtotNakMixQF.append(float(lsplit[10])/10.0)
            
##            stdevGe10eVtotNakMixQF.append(float(lsplit[11])/10.0)
##            stdevGe100eVtotNakMixQF.append(float(lsplit[12])/10.0)
##            stdevGe50eVtotNakMixQF.append(float(lsplit[13])/10.0)
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
            
            meanGe10eVeeNak.append(float(lsplit[2])/10.0)
            meanGe100eVeeNak.append(float(lsplit[3])/10.0)
            meanGe50eVeeNak.append(float(lsplit[4])/10.0)
            
##            stdevGe10eVeeNak.append(float(lsplit[5])/10.0)
##            stdevGe100eVeeNak.append(float(lsplit[6])/10.0)
##            stdevGe50eVeeNak.append(float(lsplit[7])/10.0)
            
            meanGe10eVthNak.append(float(lsplit[8])/10.0)
            meanGe100eVthNak.append(float(lsplit[9])/10.0)
            meanGe50eVthNak.append(float(lsplit[10])/10.0)
            
##            stdevGe10eVthNak.append(float(lsplit[11])/10.0)
##            stdevGe100eVthNak.append(float(lsplit[12])/10.0)
##            stdevGe50eVthNak.append(float(lsplit[13])/10.0)   #.split("\n")[0]

            meanGe10eVtotNak.append(float(lsplit[8])/10.0)
            meanGe100eVtotNak.append(float(lsplit[9])/10.0)
            meanGe50eVtotNak.append(float(lsplit[10])/10.0)
            
##            stdevGe10eVtotNak.append(float(lsplit[11])/10.0)
##            stdevGe100eVtotNak.append(float(lsplit[12])/10.0)
##            stdevGe50eVtotNak.append(float(lsplit[13])/10.0)
        except:
##            print(lsplit)
            continue
    file.close()

########################################################################
##                          For Sapphire                              ##
########################################################################


##for fname in fnamesNak:
##    file = open(folder_path_Sap_Nak+"/"+fname)
##    for line in file:
##        lsplit = line.split("\t")
##        try:
##            dt = float(lsplit[1])
##            timeSapNak.append(float(lsplit[0]))
##            deltSapNak.append(dt)
##            
##            meanSap50eVthNak.append(float(lsplit[2])/10)
##            meanSap100eVthNak.append(float(lsplit[3])/10)
##            
####            stdevSap50eVthNak.append(float(lsplit[4])/10)
####            stdevSap100eVthNak.append(float(lsplit[5])/10)
##            
##        except:
####            print(lsplit)
####            print("reading from "+folder_path+"Ge/"+fname)
##            continue
##    file.close()
##
##for fname in fnamesLiv:
##    file = open(folder_path_Sap_Liv+"/"+fname)
##    for line in file:
##        lsplit = line.split("\t")
##        try:
##            dt = float(lsplit[1])
##            timeSapLiv.append(float(lsplit[0]))
##            deltSapLiv.append(dt)
##            
##            meanSap50eVthLiv.append(float(lsplit[2])/10)
##            meanSap100eVthLiv.append(float(lsplit[3])/10)
##            
####            stdevSap50eVthLiv.append(float(lsplit[4])/10)
####            stdevSap100eVthLiv.append(float(lsplit[5])/10)
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
##plt.yscale("log")
##plt.ylim(0.1,255)
plt.xlim(0.0,1)
##plt.errorbar(timeGeNak, meanGe10eVeeNak, yerr=stdevGe10eVeeNak, label=r'Nakazato')
##plt.errorbar(timeGeLiv, meanGe10eVeeLiv, yerr=stdevGe10eVeeLiv, label=r'Livermore')
##plt.plot(timeGeNak, meanGe100eVeeNak, color='orange', label=r'Nakazato, Lindhard QF')
##plt.plot(timeGeNakMixQF, meanGe100eVeeNakMixQF, color='red', label=r'Nakazato, Modified QF')
##plt.plot(timeGeNak, meanGe50eVeeNak, color='orange', label=r'Nakazato $E_{th}=50eV_{ee}$')
plt.plot(timeGeNak, meanGe50eVeeNak, color='orange', label=r'Nakazato $t_{rev}=100ms$')
plt.plot(timeGeNakMixQF, meanGe50eVeeNakMixQF, color='blue', label=r'Nakazato $t_{rev}=300ms$',linestyle='--')
##plt.plot(timeGeNak, meanGe10eVthNak, color='orange', label=r'Nakazato')
##plt.plot(timeSapNak, meanSap100eVthNak, color='orange', label=r'Nakazato $t_{rev}=100ms$')#label=r'Nakazato $E_{th}=100eV_{t}$'
##plt.plot(timeGeLiv, meanGe100eVeeLiv, color='blue', label=r'Livermore')
##plt.plot(timeGeLiv, meanGe10eVthLiv, color='blue', label=r'Livermore')
##plt.plot(timeGeLiv, meanGe50eVeeLiv, color='blue', label=r'Livermore $E_{th}=50eV_{ee}$',linestyle='--')
##plt.plot(timeSapLiv, meanSap100eVthLiv, color='blue', label=r'Nakazato $t_{rev}=300ms$',linestyle='--')#label=r'Livermore $E_{th}=100eV_{t}$'
plt.fill_between(timeGeNak, np.array(meanGe50eVeeNak)+np.array(np.sqrt(meanGe50eVeeNak)), np.array(meanGe50eVeeNak)-np.array(np.sqrt(meanGe50eVeeNak)), color='orange', alpha=0.3)#, label=r'Nakazato 1$\sigma$'
plt.fill_between(timeGeNakMixQF, np.array(meanGe50eVeeNakMixQF)+np.array(np.sqrt(meanGe50eVeeNakMixQF)), np.array(meanGe50eVeeNakMixQF)-np.array(np.sqrt(meanGe50eVeeNakMixQF)), color='blue', alpha=0.3)
##plt.fill_between(timeGeLiv, np.array(meanGe50eVeeLiv)+np.array(np.sqrt(meanGe50eVeeLiv)), np.array(meanGe50eVeeLiv)-np.array(np.sqrt(meanGe50eVeeLiv)), color='blue', alpha=0.3)#, label=r'Livermore 1$\sigma$'
##plt.fill_between(timeGeNak, np.array(meanGe10eVthNak)+np.array(stdevGe10eVthNak), np.array(meanGe10eVthNak)-np.array(stdevGe10eVthNak), label=r'Nakazato 1$\sigma$', color='orange', alpha=0.3)
##plt.fill_between(timeGeLiv, np.array(meanGe10eVthLiv)+np.array(stdevGe10eVthLiv), np.array(meanGe10eVthLiv)-np.array(stdevGe10eVthLiv), label=r'Livermore 1$\sigma$', color='blue', alpha=0.3)
##plt.fill_between(timeSapNak, np.array(meanSap100eVthNak)+np.array(np.sqrt(meanSap100eVthNak)), np.array(meanSap100eVthNak)-np.array(np.sqrt(meanSap100eVthNak)), color='orange', alpha=0.3)#, label=r'Nakazato 1$\sigma$')
##plt.fill_between(timeSapLiv, np.array(meanSap100eVthLiv)+np.array(np.sqrt(meanSap100eVthLiv)), np.array(meanSap100eVthLiv)-np.array(np.sqrt(meanSap100eVthLiv)), color='blue', alpha=0.3)#, label=r'Livermore 1$\sigma$')
plt.xlabel("Time (in s)")
plt.ylabel("Counts/s")
plt.tick_params(axis='both', which='major')
##plt.errorbar(timeGe, meanGe100eVee, yerr=stdevGe100eVee, label=r'threshold = 100eVee')
##plt.errorbar(timeGe, meanGe50eVee, yerr=stdevGe50eVee, label=r'threshold = 50eVee')
plt.legend()
plt.show()

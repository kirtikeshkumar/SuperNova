# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:30:11 2024
Code for plotting SNn Spectra
@author: Kirtikesh Kumar
"""
import numpy as np

time=[]
nue = []
anue = []
nux = []
lnue = []
lanue = []
lnux = []
fname="intp2001"
with open(fname+".data", 'r') as file:
    for i,v in enumerate(file.readlines()):
       
        if(i%22==0):
            nuesum = 0
            anuesum= 0
            nuxsum = 0
            lnuesum = 0
            lanuesum= 0
            lnuxsum = 0
            time.append(float(v.split()[0]))
        elif(i%22!=21):
            nuesum+=float(v.split()[2])
            anuesum+=float(v.split()[3])
            nuxsum+=float(v.split()[4])
            lnuesum+=float(v.split()[5])
            lanuesum+=float(v.split()[6])
            lnuxsum+=float(v.split()[7])
        elif(i%22==21):
            nue.append(nuesum)
            anue.append(anuesum)
            nux.append(nuxsum)
            lnue.append(lnuesum)
            lanue.append(lanuesum)
            lnux.append(lnuxsum)

##Since the last sum may not have been appended
if(len(time)-1==len(nue)):
    nue.append(nuesum)
    anue.append(anuesum)
    nux.append(nuxsum)
    lnue.append(lnuesum)
    lanue.append(lanuesum)
    lnux.append(lnuxsum)
    
np.savetxt(fname+"_Flux_Luminosity.dat",np.transpose((time,nue,anue,nux,lnue,lanue,lnux)),delimiter='\t')
#np.savetxt(fname+"_Luminosity.dat",np.transpose((time,lnue,lanue,lnux)),delimiter='\t')

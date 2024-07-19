import numpy as np

time=[]
elo=[]
ehi=[]
nue = []
anue = []
nux = []
lnue = []
lanue = []
lnux = []
fname="intp2001"

with open("data/"+fname+".data", 'r') as file:
    for i,v in enumerate(file.readlines()):
       
        if(i%22==0):
            nuesum = 0
            anuesum= 0
            nuxsum = 0
            lnuesum = 0
            lanuesum= 0
            lnuxsum = 0
            time = float(v.split()[0])
            if(time-round(time,4) != 0.0):
                print(round(time,4),time)
            wfname = "data/"+fname+"/"+fname+"_"+str(round(time,4))+".dat"
        elif(i%22!=21):
            if(float(v.split()[0])!=0.0):
                elo.append(float(v.split()[0]))
            else:
                elo.append(1.0e-7)
            ehi.append(float(v.split()[1]))
            nue.append(float(v.split()[2]))
            anue.append(float(v.split()[3]))
            nux.append(float(v.split()[4]))
            lnue.append(float(v.split()[5]))
            lanue.append(float(v.split()[6]))
            lnux.append(float(v.split()[7]))
        elif(i%22==21):
            np.savetxt(wfname,np.transpose((elo,ehi,nue,anue,nux,lnue,lanue,lnux)),delimiter='\t')
            elo.clear()
            ehi.clear()
            nue.clear()
            anue.clear()
            nux.clear()
            lnue.clear()
            lanue.clear()
            lnux.clear()

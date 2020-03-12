import numpy as np
import gzip
from netCDF4 import Dataset
import pandas as pd
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from datetime import timedelta, datetime
from vard import var2d,var3d

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


rutael = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/2018YLSANTSERIES/'
rutaep = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/2018YPSANTSERIES/'
rutapr = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/'


########################################################################################################

path_data_inun = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/eventos_inundacion_sant.xlsx'
data_inun = pd.read_excel(path_data_inun)
fec_inun = data_inun['Fecha']


########################################################################################################


erasl = np.sort(glob.glob(rutael+'*.nc'))
erasp = np.sort(glob.glob(rutaep+'*.nc'))


samplel = np.array(Dataset(erasl[0]).variables['u10'][:,0,0])

########################################################################################################
matriz_eral = np.zeros([len(samplel),len(erasl)])*np.nan
namesl = []
########################################################################################################

for i in range(len(erasl)):
    datat = Dataset(erasl[i])
    vari = list(datat.variables.keys())[-1]

    data = np.array(datat.variables[vari][:,0,0])
    matriz_eral[:,i] = data
    namesl.append(vari)
    
namesl = np.array(namesl)


#print(matriz_eral.shape)
#asdas

#samplep = np.array(Dataset(erasl[0]).variables['u10'][:,0,0])

samplep = Dataset(erasp[0])
samplepl = samplep.variables['level'][:]
samplepl[-2] = 200.
samplepl[-3] = 300.

########################################################################################################
matriz_erap = np.zeros([len(samplel),len(erasp)*len(samplepl)])*np.nan
namesp = []
########################################################################################################

for i in range(len(erasp)):
    datat = Dataset(erasp[i])
    vari = list(datat.variables.keys())[-1]
    levs = datat.variables['level'][:]
    data = np.array(datat.variables[vari][:,:,0,0])
    data2 = data.copy()
    #print(data.shape)

    l1 = levs[-2]
    l2 = levs[-3]

    d1 = data2[:,-2]
    d2 = data2[:,-3]

    #print('')
    #print(d1[0])
    #print(d2[0])
    #print('')
    
    levs[-2] = l2
    levs[-3] = l1



    data[:,-2] = d2 
    data[:,-3] = d1


    oj = (len(levs)-1)*i

    for j in range(len(levs)):

        if len(vari) == 1:

            if len(str(levs[j])[:-2])==4:
                name = vari+vari+str(levs[j])[:-2]
            else:
                name = vari+vari+'0'+str(levs[j])[:-2]
                
                
        else:
            if len(str(levs[j])[:-2])==4:
                
                name = vari+str(levs[j])[:-2]
            else:
                name = vari+'0'+str(levs[j])[:-2]
                
            
        namesp.append(name)
        
        ic = i+j + oj
        
        dato = data[:,j]
        
        matriz_erap[:,ic] = dato


namesp = np.array(namesp)

#print(namesp)
#print(matriz_erap)
#print(matriz_erap.shape)


#plt.plot(matriz_erap[0,18:18+18],samplepl)
#plt.show()

#asdas
#asdasd





pre = Dataset(rutapr+'prce_santserie_2018.nc')
pri = Dataset(rutapr+'prci_santserie_2018.nc')

dates = pd.date_range('2018-01-01 00:00:00','2018-12-31 23:45:00',freq='1h')

########################################################################################################

rain_era = pd.DataFrame(pre.variables['tp'][:,0,0]*1000.,index=dates,columns=['ppt'])
rain_ime = pd.DataFrame(pri.variables['precipitationCal'][:,0,0],index=dates,columns=['ppt'])


########################################################################################################


datafvar = pd.DataFrame(matriz_erap,index=dates,columns=namesp)
#print(datafvar)



#datafvar = pd.DataFrame(matriz_eral,index=dates,columns=namesl)

datafvar = datafvar['2018-01-01':'2018-06-01']
rain_ime = rain_ime['2018-01-01':'2018-06-01']
rain_era = rain_era['2018-01-01':'2018-06-01']

x_train,x_test,y_train,y_test = train_test_split(datafvar,rain_ime,train_size=0.8,test_size=0.2,shuffle=False)
x_train2,x_test2,y_train2,y_test2 = train_test_split(datafvar,rain_era,train_size=0.8,test_size=0.2,shuffle=False)
#x_train,x_test,y_train,y_test = train_test_split(datafvar,rain_ime,train_size=0.8,test_size=0.2,shuffle=False)

regressor = DecisionTreeRegressor(max_depth=150)
regressor.fit(x_train, y_train)

tree = regressor.predict(x_test)

tree = pd.DataFrame(tree,index=y_test.index)

plt.plot(tree,lw=0.7)
plt.plot(y_test,lw=0.7)
plt.plot(y_test2,lw=0.7)

impo = regressor.feature_importances_
am = np.argmax(impo)

impof = pd.DataFrame(impo,index=namesp,columns=['impo'])
impof = impof.sort_values(by='impo')

#print(impof)

ind = impof.index
impoff = np.array(impof)
#print(ind)
for i in range(len(impof)):
    print(ind[i],impoff[i])
#print(am)
#print(namesp[am])

plt.show()


##for i in range(len(fec_inun)):
##    fec1 = fec_inun[i]
##    fec2 = fec_inun[i] + timedelta(hours=24)
##    #plt.axvline(fec1,lw = 0.8,color='red')
##    plt.axvline(fec2,lw = 0.8,color='red')
##
##plt.show()



#print(x_train)



#print(datafvar)


#rain_ime
#print(rain_ime)
#rain = rain_era-rain_ime
#plt.plot(rain_era.resample('D').sum(),lw=0.5,color='k')
#plt.plot(rain_ime.resample('D').sum(),lw=0.5,color='g')

#for i in range(len(fec_inun)):
#    fec1 = fec_inun[i]
#    fec2 = fec_inun[i] + timedelta(hours=24)
    #plt.axvline(fec1,lw = 0.8,color='red')
#    plt.axvline(fec2,lw = 0.8,color='red')
    


#plt.plot(rain)

#plt.show()

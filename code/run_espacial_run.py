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

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error,median_absolute_error

import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle as pk
from scipy.stats import pearsonr


rutaele = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/2018YLSANT/'
rutaepe = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/2018YPSANT/'
rutapr = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/'


path_data_inun = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/eventos_inundacion_sant3.xlsx'
data_inun = pd.read_excel(path_data_inun)
fec_inun = data_inun['Fecha']


##rdnn = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/dnn2/'
##dnn = keras.models.load_model(rdnn+'model_1')



def min_max(a):
    minn = np.min(a)
    maxx = np.max(a)

    norm = (a-minn)/(maxx-minn)
    
    return norm


###########################################################################3

erasl = np.sort(glob.glob(rutaele+'*.nc'))
erasp = np.sort(glob.glob(rutaepe+'*.nc'))

samplel = np.array(Dataset(erasl[0]).variables['u10'][:,:,:])

ntie = samplel.shape[0]
nlat = samplel.shape[1]
nlon = samplel.shape[2]


nmodels = nlat*nlon

ncarasl = 29
ncarasp = 162

t = fec_inun[3]


t1 = t+ timedelta(hours=5)
fd_evento = str(t1)[0:10]
t2 = t + timedelta(hours=48)

datar = pd.date_range(str(t1),str(t2),freq='1h')

print(t1,t2)

mat_total = np.zeros([len(datar),nlat*nlon])*np.nan

dates = pd.date_range('2018-01-01 00:00:00','2018-12-31 23:45:00',freq='1h')


don1 = np.where(dates==t1)[0][0]
don2 = np.where(dates==t2)[0][0]

ntie2 = len(datar)



for i in range(nmodels):
    
    matriz_eral = np.zeros([len(datar),ncarasl])*np.nan
    namesl = []
    ########################################################################################################

    for k in range(len(erasl)):
        datat = Dataset(erasl[k])
        vari = list(datat.variables.keys())[-1]



        data2 = np.array(datat.variables[vari][:,:,:])

        data = np.zeros([ntie,nlat,nlon])*np.nan
        
        for dd in range(8760):
            data[dd,:,:] = np.flipud(data2[dd,:,:])
            
        
        data = data.reshape([ntie,nlat*nlon])
        data = data[don1:don2+1,i]

        matriz_eral[:,k] = data
        namesl.append(vari)


        
    namesl = np.array(namesl)

    

    samplep = Dataset(erasp[0])
    samplepl = samplep.variables['level'][:]
    samplepl[-2] = 200.
    samplepl[-3] = 300.

    ########################################################################################################
    matriz_erap = np.zeros([ntie2,ncarasp])*np.nan
    namesp = []
    ########################################################################################################

    for h in range(len(erasp)):
        datat = Dataset(erasp[h])
        vari = list(datat.variables.keys())[-1]
        levs = datat.variables['level'][:]
        
        data2 = np.array(datat.variables[vari][:,:,:,:])


        data = np.zeros([ntie,len(levs),nlat,nlon])*np.nan
        
        
        for ddd in range(8760):
            for www in range(len(levs)):
                data[ddd,www,:,:] = np.flipud(data2[dd,www,:,:])


        data = data.reshape([ntie,len(levs),nlat*nlon])
        data = data[don1:don2+1,:,i]
        data2 = data.copy()

        l1 = levs[-2]
        l2 = levs[-3]

        d1 = data2[:,-2]
        d2 = data2[:,-3]

        
        levs[-2] = l2
        levs[-3] = l1



        data[:,-2] = d2 
        data[:,-3] = d1


        oj = (len(levs)-1)*h

        for h1 in range(len(levs)):

            if len(vari) == 1:

                if len(str(levs[h1])[:-2])==4:
                    name = vari+vari+str(levs[h1])[:-2]
                else:
                    name = vari+vari+'0'+str(levs[h1])[:-2]
                    
                    
            else:
                if len(str(levs[h1])[:-2])==4:
                    
                    name = vari+str(levs[h1])[:-2]
                else:
                    name = vari+'0'+str(levs[h1])[:-2]
                    
                
            namesp.append(name)
            
            ic = h+h1 + oj
            
            dato = data[:,h1]
            
            matriz_erap[:,ic] = dato



##
    pri = Dataset(rutapr+'prc_sant_remap_2018.nc')
##
    dates = pd.date_range('2018-01-01 00:00:00','2018-12-31 23:45:00',freq='1h')
    dates = dates[don1:don2+1]
##
    pri_pri = pri.variables['precipitationCal'][don1:don2+1,:,:]
    pri_pri = pri_pri.reshape([ntie2,nlat*nlon])
    pri_pri = pri_pri[:,i]

    pri_min = np.min(pri_pri)
    pri_max = np.max(pri_pri)
    


##
##    pri_me = pri_pri.mean(axis=1)



    #rain_ime = pd.DataFrame(pri_pri[:,i],index=dates,columns=['ppt'])


    datafvarp = pd.DataFrame(matriz_erap,index=dates,columns=namesp)
    datafvarp = datafvarp.fillna(datafvarp.mean())
    datafvarl = pd.DataFrame(matriz_eral,index=dates,columns=namesl)

    datafvarp = pd.concat([datafvarp, datafvarl], axis=1, sort=False)

    datafvarp = datafvarp.fillna(0.0)



    rsave_mod = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS_ESP/'


    if i < 10:
        i2 = '00'+str(i)
    elif i >= 10 and i <100:
        i2 = '0'+str(i)
    elif i >=100:
        i2 = str(i)

    print(i2)
    
    with open(rsave_mod+'svm_model_'+i2,'rb') as mod:
        svmre = pk.load(mod)
        sca2 = pk.load(mod)


    xs2 = datafvarp
    #sca3 = MinMaxScaler()
    #sca3.fit(datafvarp)
    #xs2 = sca3.transform(datafvarp)
        
    #xs2 = datafvarp
    #print(xs2)
    svmrep = svmre.predict(xs2)

    #svmrep =  svmrep*(pri_max-pri_min)+pri_min


    #svmrep = sca2.inverse_transform(svmrep.reshape(-1, 1))

##    plt.plot(pri_pri)
##    plt.plot(svmrep)
##    plt.show()



##    print(svmrep)
    mat_total[:,i] = np.array(svmrep)


print(mat_total.shape)


with open('/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/mat_total3','wb') as tot:
    pk.dump(mat_total,tot)



    








    

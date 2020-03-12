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
from sklearn.preprocessing import MinMaxScaler
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


rdnn = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/dnn2/'
dnn = keras.models.load_model(rdnn+'model_1')





###########################################################################3



########################################################################################################


erasl = np.sort(glob.glob(rutaele+'*.nc'))
erasp = np.sort(glob.glob(rutaepe+'*.nc'))


t = fec_inun[10]

t1 = t+ timedelta(hours=5)
fd_evento = str(t1)[0:10]
t2 = t + timedelta(hours=48)

datar = pd.date_range(str(t1),str(t2),freq='1h')

print(t1,t2)

dates = pd.date_range('2018-01-01 00:00:00','2018-12-31 23:45:00',freq='1h')


samplel = np.array(Dataset(erasl[0]).variables['u10'][0,:,:])
ncarasl = 29
ncarasp = 162


for i,t in enumerate(datar):

    ########################################################################################################
    matriz_eral = np.zeros([samplel.shape[0]*samplel.shape[1],ncarasl])*np.nan
    namesl = []
    ########################################################################################################
    don = np.where(dates==t)[0][0]
    
    for l in range(len(erasl)):
        datat = Dataset(erasl[l])
        vari = list(datat.variables.keys())[-1]
        
        data = np.array(datat.variables[vari][don,:,:])
        
        data = data.reshape(-1)

        matriz_eral[:,l] = data
        namesl.append(vari)

        



    samplep = Dataset(erasp[0])
    samplepl = samplep.variables['level'][:]
    samplepl[-2] = 200.
    samplepl[-3] = 300.

    ########################################################################################################
    matriz_erap = np.zeros([samplel.shape[0]*samplel.shape[1],ncarasp])*np.nan
    namesp = []
    ########################################################################################################
    
    for z in range(len(erasp)):
        datat = Dataset(erasp[z])
        
        vari = list(datat.variables.keys())[-1]
        
        levs = datat.variables['level'][:]
        
        data = np.array(datat.variables[vari][don,:,:,:])



        data2 = data.copy()


        l1 = levs[-2]
        l2 = levs[-3]

        d1 = data2[-2,:,:]
        d2 = data2[-3,:,:]

        
        levs[-2] = l2
        levs[-3] = l1


        data[-2,:,:] = d2 
        data[-2,:,:] = d1



        oj = (len(levs)-1)*z


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
            
            ic = z+j + oj
            
            dato = data[j,:,:].reshape(-1)

##            
            matriz_erap[:,ic] = dato
            


    datafvarp = pd.DataFrame(matriz_erap,columns=namesp)
    print(datafvarp)
    datafvarp = datafvarp.fillna(datafvarp.mean())
    datafvarl = pd.DataFrame(matriz_eral,columns=namesl)


    datafvarp = pd.concat([datafvarp, datafvarl], axis=1, sort=False)

#############################################################################################################

    pri = Dataset(rutapr+'prc_sant_remap_2018.nc')
    rain_ime = pri.variables['precipitationCal'][don,:,:]

    with open('/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/scalador','rb') as con:
        scaa1 = pk.load(con)
        scaa2 = pk.load(con)
        scaa3=  pk.load(con)


    xs2 = scaa3.transform(datafvarp.values)
    
    dnnregp = dnn.predict(xs2)
    dnnregp = scaa2.inverse_transform(dnnregp)

    plt.plot(rain_ime.reshape(-1))
    plt.plot(dnnregp)
    plt.show()

    rain_dnn = dnnregp.reshape([16,14])
    #rain_dnn[rain_dnn<0]=0

    print(rain_dnn.shape)
    print(rain_ime.shape)


    

    
    plt.figure(figsize=(10,3))
    plt.subplot(121)
    plt.imshow(rain_ime)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(rain_dnn)
    plt.colorbar()
    
    plt.show()
    

        

##
##        data = np.array(datat.variables[vari][:,0,0])
##        matriz_eral[:,i] = data
##        namesl.append(vari)
##        
##    namesl = np.array(namesl)
##
##
##    samplep = Dataset(erasp[0])
##    samplepl = samplep.variables['level'][:]
##    samplepl[-2] = 200.
##    samplepl[-3] = 300.






    

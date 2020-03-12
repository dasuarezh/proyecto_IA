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

import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle as pk



rutael = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/2018YLSANTSERIES/'
rutaep = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/2018YPSANTSERIES/'
rutapr = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/'


########################################################################################################

path_data_inun = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/eventos_inundacion_sant3.xlsx'
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

    l1 = levs[-2]
    l2 = levs[-3]

    d1 = data2[:,-2]
    d2 = data2[:,-3]

    
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



pre = Dataset(rutapr+'prce_santserie_2018.nc')
pri = Dataset(rutapr+'prci_santserie_2018.nc')

dates = pd.date_range('2018-01-01 00:00:00','2018-12-31 23:45:00',freq='1h')

########################################################################################################

rain_era = pd.DataFrame(pre.variables['tp'][:,0,0]*1000.,index=dates,columns=['ppt'])
rain_ime = pd.DataFrame(pri.variables['precipitationCal'][:,0,0],index=dates,columns=['ppt'])


########################################################################################################


datafvarp = pd.DataFrame(matriz_erap,index=dates,columns=namesp)
datafvarl = pd.DataFrame(matriz_eral,index=dates,columns=namesl)


datafvarp = pd.concat([datafvarp, datafvarl], axis=1, sort=False)



datafvarp2 = datafvarp.copy()
rain_ime2 = rain_ime.copy()

#print(datafvarp2)




for i,t in enumerate(fec_inun):

    t1 = t+ timedelta(hours=5)
    fd_evento = str(t1)[0:10]
    #muni = fec_mun[i].replace(' ','_')
    #print(fd_evento[0:10])
    
    t2 = t + timedelta(hours=48)

    datar = pd.date_range(str(t1),str(t2),freq='1h')

    print(t1,t2)

    
    datafvarp2 = datafvarp2.drop(datar)
    rain_ime2 = rain_ime2.drop(datar)



print(datafvarp2.shape,rain_ime2.shape)


rsave_mod = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/'

for i,t in enumerate(fec_inun):

    t1 = t+ timedelta(hours=5)
    fd_evento = str(t1)[0:10]
    #muni = fec_mun[i].replace(' ','_')
    #print(fd_evento[0:10])
    
    t2 = t + timedelta(hours=48)


    print(t1,t2)


    datafvarp3 = datafvarp[t1:t2]
    rain_ime3 = rain_ime[t1:t2]

##    # MLR
##    #####################################################################
    mlreg = LinearRegression()

    mlreg.fit(datafvarp2, rain_ime2)


    with open(rsave_mod+'mlr_model','wb') as mod:
        pk.dump(mlreg,mod)

    print('mlr_trained')
    

##    mlregp = mlreg.predict(datafvarp3)
##    mlregpf = pd.DataFrame(mlregp,index=rain_ime3.index)
##    

##    # KNN
##    #####################################################################
    knnreg = KNeighborsRegressor(weights='distance')

    knnreg.fit(datafvarp2, rain_ime2)

    with open(rsave_mod+'knn_model','wb') as mod:
        pk.dump(knnreg,mod)

    print('knn_trained')


##    knnregp = knnreg.predict(datafvarp3.values)
##    knnregpf = pd.DataFrame(knnregp,index=rain_ime3.index)
 
    
##    # DECISION TREE
##    #####################################################################
    dtreg = DecisionTreeRegressor(max_depth=30)
    dtreg.fit(datafvarp2, rain_ime2)


    with open(rsave_mod+'dtr_model','wb') as mod:
        pk.dump(dtreg,mod)

    print('dtr_trained')

##    dtregp = dtreg.predict(datafvarp3.values)
##    dtregpf = pd.DataFrame(dtregp,index=rain_ime3.index)
##    #####################################################################
##    
##
##    # RFR
##    #####################################################################
    rfreg = RandomForestRegressor(n_estimators=10)
    rfreg.fit(datafvarp2, rain_ime2)


    with open(rsave_mod+'rft_model','wb') as mod:
        pk.dump(rfreg,mod)

    print('rft_trained')
##    rfregp = rfreg.predict(datafvarp3.values)
##    rfregpf = pd.DataFrame(rfregp,index=rain_ime3.index)
##    #####################################################################
##
####    # SVM
####    #####################################################################
    svmre = SVR()

    sca1 = MinMaxScaler()
    sca2 = MinMaxScaler()
    sca3 = MinMaxScaler()

    sca1.fit(datafvarp2)
    sca2.fit(rain_ime2)
    sca3.fit(datafvarp3.values)

    xs = sca1.transform(datafvarp2)
    ys = sca2.transform(rain_ime2)
    xs2 = sca3.transform(datafvarp3.values)
    
    svmre.fit(xs,ys)

    

    with open(rsave_mod+'svm_model','wb') as mod:
        pk.dump(svmre,mod)
        
    print('svm_trained')


    break
##    svmrep = svmre.predict(xs2)
##
##    svmrep = sca2.inverse_transform(svmrep.reshape(-1, 1))
##    svmrepf = pd.DataFrame(svmrep,index=rain_ime3.index)
####    #####################################################################
##
##
##

    

    # DNN
    #####################################################################
    
##    dnn = keras.models.Sequential([
##        keras.layers.Dense(162, activation=tf.nn.relu, input_shape=[162]),
##        keras.layers.Dense(50, activation=tf.nn.relu),
##        keras.layers.Dense(1)
##        ])
##
##    xs = preprocessing.scale(datafvarp2)
##    ys = preprocessing.scale(rain_ime2)
##    #ys2 = sc_y.fit_transform(rain_ime2)
##
##    xs2 = preprocessing.scale(datafvarp3.values)
##
##    optimizer = tf.keras.optimizers.RMSprop(0.0099)
##    
##    dnn.compile(loss='mean_squared_error',
##                optimizer=optimizer,
##                metrics=['mae', 'mse'])
##
##    dnn.fit(xs, ys,batch_size=10,epochs=15)
##
##
##    dnnregp = dnn.predict(xs2)
##    dnnregpf = pd.DataFrame(dnnregp,index=rain_ime3.index)
    #####################################################################

    
    #print(Dtregp)

##    plt.figure(figsize=[10,4])
##    plt.plot(rain_ime3,color='k')
##    plt.plot(dtregpf,color='b',ls='--')
##    plt.plot(svmrepf,color='g',ls='--')
##    plt.plot(mlregpf,color='r',ls='--')
##    plt.plot(knnregpf,color='magenta',ls='--')
##    plt.plot(rfregpf,color='orange',ls='--')
####    plt.plot(dnnregpf,color='cyan')
##    
##    plt.show()
    





    




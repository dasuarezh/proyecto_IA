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

#





##################################################################################

rsave_mod = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/'


with open(rsave_mod+'mlr_model','rb') as mod:
    mlreg = pk.load(mod)


with open(rsave_mod+'knn_model','rb') as mod:
    knnreg = pk.load(mod)


with open(rsave_mod+'dtr_model','rb') as mod:
    dtreg = pk.load(mod)


with open(rsave_mod+'rft_model','rb') as mod:
    rfreg = pk.load(mod)


with open(rsave_mod+'svm_model','rb') as mod:
    svmre = pk.load(mod)


rdnn = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/dnn2/'
dnn = keras.models.load_model(rdnn+'model_1')




error_mat = np.zeros([5,14])*np.nan


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
    

##    mlregp = mlreg.predict(datafvarp3)
##    mlregpf = pd.DataFrame(mlregp,index=rain_ime3.index)

##    

##    # KNN
##    #####################################################################


    knnregp = knnreg.predict(datafvarp3.values)
    knnregpf = pd.DataFrame(knnregp,index=rain_ime3.index)
 
    
##    # DECISION TREE
##    #####################################################################

    dtregp = dtreg.predict(datafvarp3.values)
    dtregpf = pd.DataFrame(dtregp,index=rain_ime3.index)
    
##    #####################################################################
##    
##
##    # RFR
##    #####################################################################

    rfregp = rfreg.predict(datafvarp3.values)
    rfregpf = pd.DataFrame(rfregp,index=rain_ime3.index)
    
##    #####################################################################
##
####    # SVM
####    #####################################################################

    

    sca2 = MinMaxScaler()
    sca3 = MinMaxScaler()


    sca2.fit(rain_ime2)
    sca3.fit(datafvarp3.values)


    ys = sca2.transform(rain_ime2)
    xs2 = sca3.transform(datafvarp3.values)
    
    svmrep = svmre.predict(xs2)
    svmrep = sca2.inverse_transform(svmrep.reshape(-1, 1))
    svmrepf = pd.DataFrame(svmrep,index=rain_ime3.index)
    
####    #####################################################################

    # DNN
    #####################################################################


    scaa1 = MinMaxScaler()
    scaa2 = MinMaxScaler()
    scaa3 = MinMaxScaler()

    scaa1.fit(datafvarp2)
    scaa2.fit(rain_ime2)
    scaa3.fit(datafvarp3.values)

##    with open('/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/scalador','wb') as con:
##        pk.dump(scaa1,con)
##        pk.dump(scaa2,con)
##        pk.dump(scaa3,con)
##
##    asdasd

    xs = scaa1.transform(datafvarp2)
    ys = scaa2.transform(rain_ime2)
    xs2 = scaa3.transform(datafvarp3.values)

    
    dnnregp = dnn.predict(xs2)
    dnnregp = sca2.inverse_transform(dnnregp)
    dnnregpf = pd.DataFrame(dnnregp,index=rain_ime3.index)
    #####################################################################
    



########    mlr_err = mean_squared_error(rain_ime3,mlregp)



    
##    knn_err = mean_squared_error(rain_ime3,knnregp)
##    dtr_err = mean_squared_error(rain_ime3,dtregp)
##    rfr_err = mean_squared_error(rain_ime3,rfregp)
##    svm_err = mean_squared_error(rain_ime3,svmrep)
##    dnn_err = mean_squared_error(rain_ime3,dnnregp)


##    mlr_err = median_absolute_error(rain_ime3,mlregp)
##    knn_err = median_absolute_error(rain_ime3,knnregp)
##    dtr_err = median_absolute_error(rain_ime3,dtregp)
##    rfr_err = median_absolute_error(rain_ime3,rfregp)
##    svm_err = median_absolute_error(rain_ime3,svmrep)
##    dnn_err = median_absolute_error(rain_ime3,dnnregp)
    
    
    #print(np.array(rain_ime3)[:,0])
    #print(np.array(mlregp)[:,0])

    
##########    mlr_err = pearsonr(np.array(rain_ime3)[:,0],np.array(mlregp)[:,0])[0]




    knn_err = pearsonr(np.array(rain_ime3)[:,0],np.array(knnregp)[:,0])[0]

    dtr_err = pearsonr(np.array(rain_ime3)[:,0],np.array(dtregp))[0]

    rfr_err = pearsonr(np.array(rain_ime3)[:,0],np.array(rfregp))[0]

    svm_err = pearsonr(np.array(rain_ime3)[:,0],np.array(svmrep)[:,0])[0]

    dnn_err = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp)[:,0])[0]





    #errores = np.array([mlr_err,knn_err,dtr_err,rfr_err,svm_err,dnn_err])
    errores = np.array([knn_err,dtr_err,rfr_err,svm_err,dnn_err])

    print(errores)


    error_mat[:,i] = errores


    #####################################################################
    
    #print(Dtregp)

    rsave_img = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/PREDICTION_SERIES/'
    rsave_img_er = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/ERRORES/'

##    fig = plt.figure(figsize=[10,4])
##    plt.ylabel('Precipitacion [mm/h]')
##
##
##
##    plt.plot(rain_ime3,color='k',label='IMERG',lw=1.1)
##########    plt.plot(mlregpf,color='r',ls='--',lw=0.9,label='MultipleLinearR')
##    plt.plot(knnregpf,color='magenta',ls='--',lw=0.9,label='KNN')
##    plt.plot(dtregpf,color='b',ls='--',lw=0.9,label='DesicionTree')
##    plt.plot(rfregpf,color='orange',ls='--',lw=0.9,label='RandomForest')
##    plt.plot(svmrepf,color='g',ls='--',lw=0.9,label='SupportVectorMachine')
##    plt.plot(dnnregpf,color='cyan',ls='--',lw=0.9,label='DNN')
##
##    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=4,fontsize=9.6)
##
##
##
##    lef  = 0.13
##    botto = 0.24
##    righ = 0.90
##    to = 0.95
##    wspac = 0.21
##    hspac = 0.20
##
##
##    fig.subplots_adjust(left=lef, bottom=botto, right=righ,
##                    top= to, wspace=wspac, hspace=hspac)
##    
######    plt.plot(dnnregpf,color='cyan')
####
##    plt.savefig(rsave_img+'evento'+str(i),bbox_inches='tight',dpi=200)
##    plt.show()
##    
##
##
##plt.close()


metods= ['KNeaNei','DesicionTree','RandForest','SuppVectMach','DeepNeuNet']
events= ['01','02','03','04','05','06','07','08','09','10','11','12','13','Mean']

error_mat[:,-1] = np.mean(error_mat[:,:-1],axis=1)

print(error_mat)






    
##fig = plt.figure(figsize=[10,4])
##plt.pcolor(error_mat,cmap='gist_stern_r')
##plt.xticks(np.arange(0.5,14.0,1.0),events)
##plt.yticks(np.arange(0.5,5.0,1.0),metods[::-1])
##plt.colorbar()
##plt.savefig(rsave_img_er+'mean_sq_err_metods',bbox_inches='tight',dpi=200)
##plt.show()
##



fig = plt.figure(figsize=[10,4])
plt.pcolor(error_mat,cmap='bwr_r')
plt.xticks(np.arange(0.5,14.0,1.0),events)
plt.yticks(np.arange(0.5,5.0,1.0),metods[::-1])
plt.colorbar()
plt.savefig(rsave_img_er+'corr_metods',bbox_inches='tight',dpi=200)
plt.show()
















##
##
##### DNNS
###################################################################################
##
##
##
##
##rdnn = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/dnn2/'
##
##dnn00 = keras.models.load_model(rdnn+'model_0')
##dnn01 = keras.models.load_model(rdnn+'model_1')
##dnn02 = keras.models.load_model(rdnn+'model_2')
##dnn03 = keras.models.load_model(rdnn+'model_3')
##dnn04 = keras.models.load_model(rdnn+'model_4')
##dnn05 = keras.models.load_model(rdnn+'model_5')
##dnn06 = keras.models.load_model(rdnn+'model_6')
##dnn07 = keras.models.load_model(rdnn+'model_7')
##dnn08 = keras.models.load_model(rdnn+'model_8')
##dnn09 = keras.models.load_model(rdnn+'model_9')
##dnn10 = keras.models.load_model(rdnn+'model_10')
##dnn11 = keras.models.load_model(rdnn+'model_11')
##dnn12 = keras.models.load_model(rdnn+'model_12')
##
##
##error_mat = np.zeros([13,14])*np.nan
##
##
##for i,t in enumerate(fec_inun):
##
##
##    t1 = t+ timedelta(hours=5)
##    fd_evento = str(t1)[0:10]
##    #muni = fec_mun[i].replace(' ','_')
##    #print(fd_evento[0:10])
##    
##    t2 = t + timedelta(hours=48)
##
##
##    print(t1,t2)
##
##
##    datafvarp3 = datafvarp[t1:t2]
##    rain_ime3 = rain_ime[t1:t2]
##
##
##
##
##    scaa1 = MinMaxScaler()
##    scaa2 = MinMaxScaler()
##    scaa3 = MinMaxScaler()
##
##    scaa1.fit(datafvarp2)
##    scaa2.fit(rain_ime2)
##    scaa3.fit(datafvarp3.values)
##
##    xs = scaa1.transform(datafvarp2)
##    ys = scaa2.transform(rain_ime2)
##    xs2 = scaa3.transform(datafvarp3.values)
##
##    
##    dnnregp00 = dnn00.predict(xs2)
##    dnnregp01 = dnn01.predict(xs2)
##    dnnregp02 = dnn02.predict(xs2)
##    dnnregp03 = dnn03.predict(xs2)
##    dnnregp04 = dnn04.predict(xs2)
##    dnnregp05 = dnn05.predict(xs2)
##    dnnregp06 = dnn06.predict(xs2)
##    dnnregp07 = dnn07.predict(xs2)
##    dnnregp08 = dnn08.predict(xs2)
##    dnnregp09 = dnn09.predict(xs2)
##    dnnregp10 = dnn10.predict(xs2)
##    dnnregp11 = dnn11.predict(xs2)
##    dnnregp12 = dnn12.predict(xs2)
##
##    
##    dnnregp00 = scaa2.inverse_transform(dnnregp00)
##    dnnregp01 = scaa2.inverse_transform(dnnregp01)
##    dnnregp02 = scaa2.inverse_transform(dnnregp02)
##    dnnregp03 = scaa2.inverse_transform(dnnregp03)
##    dnnregp04 = scaa2.inverse_transform(dnnregp04)
##    dnnregp05 = scaa2.inverse_transform(dnnregp05)
##    dnnregp06 = scaa2.inverse_transform(dnnregp06)
##    dnnregp07 = scaa2.inverse_transform(dnnregp07)
##    dnnregp08 = scaa2.inverse_transform(dnnregp08)
##    dnnregp09 = scaa2.inverse_transform(dnnregp09)
##    dnnregp10 = scaa2.inverse_transform(dnnregp10)
##    dnnregp11 = scaa2.inverse_transform(dnnregp11)
##    dnnregp12 = scaa2.inverse_transform(dnnregp12)
##    
##    
##    dnnregpf00 = pd.DataFrame(dnnregp00,index=rain_ime3.index)
##    dnnregpf01 = pd.DataFrame(dnnregp01,index=rain_ime3.index)
##    dnnregpf02 = pd.DataFrame(dnnregp02,index=rain_ime3.index)
##    dnnregpf03 = pd.DataFrame(dnnregp03,index=rain_ime3.index)
##    dnnregpf04 = pd.DataFrame(dnnregp04,index=rain_ime3.index)
##    dnnregpf05 = pd.DataFrame(dnnregp05,index=rain_ime3.index)
##    dnnregpf06 = pd.DataFrame(dnnregp06,index=rain_ime3.index)
##    dnnregpf07 = pd.DataFrame(dnnregp07,index=rain_ime3.index)
##    dnnregpf08 = pd.DataFrame(dnnregp08,index=rain_ime3.index)
##    dnnregpf09 = pd.DataFrame(dnnregp09,index=rain_ime3.index)
##    dnnregpf10 = pd.DataFrame(dnnregp10,index=rain_ime3.index)
##    dnnregpf11 = pd.DataFrame(dnnregp11,index=rain_ime3.index)
##    dnnregpf12 = pd.DataFrame(dnnregp12,index=rain_ime3.index)
##
##    
##    
##    #####################################################################
##    
##
##
##
##########    mlr_err = mean_squared_error(rain_ime3,mlregp)
##
##
##
##    
####    knn_err = mean_squared_error(rain_ime3,knnregp)
####    dtr_err = mean_squared_error(rain_ime3,dtregp)
####    rfr_err = mean_squared_error(rain_ime3,rfregp)
####    svm_err = mean_squared_error(rain_ime3,svmrep)
####    dnn_err = mean_squared_error(rain_ime3,dnnregp)
##
##
####    mlr_err = median_absolute_error(rain_ime3,mlregp)
####    knn_err = median_absolute_error(rain_ime3,knnregp)
####    dtr_err = median_absolute_error(rain_ime3,dtregp)
####    rfr_err = median_absolute_error(rain_ime3,rfregp)
####    svm_err = median_absolute_error(rain_ime3,svmrep)
####    dnn_err = median_absolute_error(rain_ime3,dnnregp)
##    
##    
##    #print(np.array(rain_ime3)[:,0])
##    #print(np.array(mlregp)[:,0])
##
##    
############    mlr_err = pearsonr(np.array(rain_ime3)[:,0],np.array(mlregp)[:,0])[0]
##
##
####    dnn_err00 = mean_squared_error(rain_ime3,dnnregp00)
####    dnn_err01 = mean_squared_error(rain_ime3,dnnregp01)
####    dnn_err02 = mean_squared_error(rain_ime3,dnnregp02)
####    dnn_err03 = mean_squared_error(rain_ime3,dnnregp03)
####    dnn_err04 = mean_squared_error(rain_ime3,dnnregp04)
####    dnn_err05 = mean_squared_error(rain_ime3,dnnregp05)
####    dnn_err06 = mean_squared_error(rain_ime3,dnnregp06)
####    dnn_err07 = mean_squared_error(rain_ime3,dnnregp07)
####    dnn_err08 = mean_squared_error(rain_ime3,dnnregp08)
####    dnn_err09 = mean_squared_error(rain_ime3,dnnregp09)
####    dnn_err10 = mean_squared_error(rain_ime3,dnnregp10)
####    dnn_err11 = mean_squared_error(rain_ime3,dnnregp11)
####    dnn_err12 = mean_squared_error(rain_ime3,dnnregp12)
##
##
##    dnn_err00 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp00)[:,0])[0]
##    dnn_err01 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp01)[:,0])[0]
##    dnn_err02 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp02)[:,0])[0]
##    dnn_err03 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp03)[:,0])[0]
##    dnn_err04 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp04)[:,0])[0]
##    dnn_err05 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp05)[:,0])[0]
##    dnn_err06 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp06)[:,0])[0]
##    dnn_err07 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp07)[:,0])[0]
##    dnn_err08 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp08)[:,0])[0]
##    dnn_err09 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp09)[:,0])[0]
##    dnn_err10 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp10)[:,0])[0]
##    dnn_err11 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp11)[:,0])[0]
##    dnn_err12 = pearsonr(np.array(rain_ime3)[:,0],np.array(dnnregp12)[:,0])[0]
##
##
##
##
##
##    #errores = np.array([mlr_err,knn_err,dtr_err,rfr_err,svm_err,dnn_err])
##    errores = np.array([dnn_err00,dnn_err01,dnn_err02,dnn_err03,dnn_err04,dnn_err05,dnn_err06,
##                        dnn_err07,dnn_err08,dnn_err09,dnn_err10,dnn_err11,dnn_err12])
##
##    print(errores)
##
##
##    error_mat[:,i] = errores
##
##
##    #####################################################################
##    
##    #print(Dtregp)
##
##    rsave_img = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/PRED_DNN/'
##    rsave_img_er = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/ERRORES/'
##
####    fig = plt.figure(figsize=[10,4])
####    plt.ylabel('Precipitacion [mm/h]')
####
####
####
####    plt.plot(rain_ime3,color='k',label='IMERG',lw=1.1)
############    plt.plot(mlregpf,color='r',ls='--',lw=0.9,label='MultipleLinearR')
####
####    plt.plot(dnnregpf00,ls='--',lw=0.9,label='DNN00')
####    plt.plot(dnnregpf01,ls='--',lw=0.9,label='DNN01')
####    plt.plot(dnnregpf02,ls='--',lw=0.9,label='DNN02')
####    plt.plot(dnnregpf03,ls='--',lw=0.9,label='DNN03')
####    plt.plot(dnnregpf04,ls='--',lw=0.9,label='DNN04')
####    plt.plot(dnnregpf05,ls='--',lw=0.9,label='DNN05')
####    plt.plot(dnnregpf06,ls='--',lw=0.9,label='DNN06')
####    plt.plot(dnnregpf07,ls='--',lw=0.9,label='DNN07')
####    plt.plot(dnnregpf08,ls='--',lw=0.9,label='DNN08')
####    plt.plot(dnnregpf09,ls='--',lw=0.9,label='DNN09')
####    plt.plot(dnnregpf10,ls='--',lw=0.9,label='DNN10')
####    plt.plot(dnnregpf11,ls='--',lw=0.9,label='DNN11')
####    plt.plot(dnnregpf12,ls='--',lw=0.9,label='DNN12')
####
####    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=7,fontsize=9.6)
####
####
####
####    lef  = 0.13
####    botto = 0.24
####    righ = 0.90
####    to = 0.95
####    wspac = 0.21
####    hspac = 0.20
####
####
####    fig.subplots_adjust(left=lef, bottom=botto, right=righ,
####                    top= to, wspace=wspac, hspace=hspac)
####    
########    plt.plot(dnnregpf,color='cyan')
######
####    plt.savefig(rsave_img+'evento'+str(i),bbox_inches='tight',dpi=200)
####    plt.show()
####    
####
####
####plt.close()
##
##
##metods= ['DNN00','DNN01','DNN02','DNN03','DNN04','DNN05','DNN06','DNN07','DNN08','DNN09','DNN10','DNN11','DNN12']
##events= ['01','02','03','04','05','06','07','08','09','10','11','12','13','Mean']
##
##error_mat[:,-1] = np.mean(error_mat[:,:-1],axis=1)
##
##print(error_mat)
##
##
##
##
##fig = plt.figure(figsize=[10,4])
##plt.pcolor(error_mat,cmap='bwr_r')
##plt.xticks(np.arange(0.5,14.0,1.0),events)
##plt.yticks(np.arange(0.5,13.0,1.0),metods[::-1])
##plt.colorbar()
##plt.savefig(rsave_img_er+'corr_dnns',bbox_inches='tight',dpi=200)
##plt.show()
##
##    
####fig = plt.figure(figsize=[10,4])
####plt.pcolor(error_mat,cmap='gist_stern_r')
####plt.xticks(np.arange(0.5,14.0,1.0),events)
####plt.yticks(np.arange(0.5,13.0,1.0),metods[::-1])
####plt.colorbar()
####plt.savefig(rsave_img_er+'mean_sq_err_dnns',bbox_inches='tight',dpi=200)
####plt.show()

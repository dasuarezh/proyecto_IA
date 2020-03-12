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


##rdnn = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/MODELS/dnn2/'
##dnn = keras.models.load_model(rdnn+'model_1')


def cmap_radar():
   cdict = { 'red' : [(0.0,   1.0,   1.0),
                      (0.067, 1.0,   0.0),
                      (0.133, 0.004, 0.004),
                      (0.2,   0.0,   0.0),
                      (0.467, 0.0,   1.0),
                      (0.533, 0.905, 0.905),
                      (0.6,   1.0,   1.0),
                      (0.667, 1.0,   1.0),
                      (0.733, 0.839, 0.839),
                      (0.8,   0.753, 0.753),
                      (0.867, 0.588, 1.0),
                      (0.933, 0.6,   0.6),
                      (1.0,   1.0,   1.0)],

           'green':  [(0.0,   1.0,   1.0),
                      (0.067, 0.923, 0.923),
                      (0.133, 0.627, 0.627),
                      (0.2,   0.0,   0.0),
                      (0.267, 1.0,   1.0),
                      (0.333, 0.784, 0.784),
                      (0.4,   0.6,   0.6),
                      (0.467, 0.55,  1.0),
                      (0.533, 0.753, 0.753),
                      (0.6,   0.656, 0.656),
                      (0.667, 0.0,   0.0),
                      (0.933, 0.0,   0.333),
                      (1.0,   0.333, 1.0)],

           'blue' :  [(0.0,   1.0,   1.0),
                      (0.067, 0.923, 0.923),
                      (0.133, 0.965, 0.965),
                      (0.2,   0.965, 0.965),
                      (0.267, 0.0,   0.0),
                      (0.867, 0.0,   1.0),
                      (0.933, 0.788, 0.788),
                      (1.0,   1.0,   1.0)] }


   #return matplotlib.colors.LinearSegmentedColormap('my_radar', cdict)#, lut=512)
   mpl.cm.register_cmap(name='pymeteo_radar', data=cdict, lut=512)

cmap_radar()


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

with open('/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/mat_total','rb') as tot:
    rain_svm = pk.load(tot)


rain_svm = rain_svm.reshape([44,16,14])
##rain_svm[rain_svm<0] = 0


dates = dates[don1:don2+1]
pri = Dataset(rutapr+'prc_sant_remap_2018.nc')
pri_pri = pri.variables['precipitationCal'][don1:don2+1,:,:]

preci_lat = pri['latitude'][:]
preci_lon = pri['longitude'][:]

##
##error_mat = np.zeros([44,14, 16])*np.nan
##for i in range(44):
##
##   mapa_ime = pri_pri[i,:,:].T
##   mapa_mod = np.flipud(rain_svm[i,:,:]).T
##
##   error = np.abs(mapa_ime-mapa_mod)/np.abs(mapa_ime)
##
##   error_mat[i,:,:] = error
##   
##
##er = np.mean(error_mat,axis=0)
##print(er)
##er[er>100] = np.nan
##
##plt.pcolor(er)
##plt.colorbar()
##plt.show()
##
##
##asdasd

   





for i in range(len(datar)):


    rain_ime = pri_pri[i,:,:]

    rain_mod = rain_svm[i,:,:]

    cmap1 = 'pymeteo_radar'

    bounds = np.arange(0.,10,0.05)

    plt.figure(figsize=(10,3))
    plt.subplot(121)
    
    m = Basemap(llcrnrlat=np.min(preci_lat),urcrnrlat=np.max(preci_lat),
                llcrnrlon=np.min(preci_lon),urcrnrlon=np.max(preci_lon),
                rsphere=6371200.,resolution='l',area_thresh=10000)

    lons, lats = np.meshgrid(preci_lon, preci_lat)
    x, y = m(lons, lats)
    
    rutashape = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/SHAPE/santander/'
    m.readshapefile(rutashape +'santander', 'Santander',
                    linewidth=0.9, color='gray')
    
    #plt.contourf(rain_ime,cmap= cmap1,extend='max',levels=bounds)
    m.contourf(x,y,rain_ime.T, cmap= cmap1,extend='max',levels=bounds)
    plt.colorbar()
    plt.title('IMERG')
    
    plt.subplot(122)


    m = Basemap(llcrnrlat=np.min(preci_lat),urcrnrlat=np.max(preci_lat),
                llcrnrlon=np.min(preci_lon),urcrnrlon=np.max(preci_lon),
                rsphere=6371200.,resolution='l',area_thresh=10000)

    lons, lats = np.meshgrid(preci_lon, preci_lat)
    x, y = m(lons, lats)

    m.readshapefile(rutashape +'santander', 'Santander',
                    linewidth=0.9, color='gray')

    m.contourf(x,y,np.flipud(rain_mod).T, cmap= cmap1,extend='max',levels=bounds)
    
    #plt.contourf(np.flipud(rain_mod),cmap= cmap1,extend='max',levels=bounds)
    plt.colorbar()
    plt.title('RFR')
    
    plt.show()
    if i <10:
        ii = '0'+str(i)
    elif i>=10:
        ii= str(i)

##    rfsave= '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/PRED_ESP/'
##    plt.savefig(rfsave+'event_'+ii,bbox_inches='tight',dpi=200)






    








    

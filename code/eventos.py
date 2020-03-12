import numpy as np
import gzip
from netCDF4 import Dataset,num2date
import pandas as pd
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
from datetime import timedelta, datetime
#from vard import var2d,var3d





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







#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor


rutapr = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/DATA/'

pres = Dataset(rutapr+'prce_santserie_2018.nc')
pris = Dataset(rutapr+'prci_santserie_2018.nc')

prea = Dataset(rutapr+'prc_sant_2018.nc')
pria = Dataset(rutapr+'prc_sant_remap_2018.nc')


preci = pria['precipitationCal'][:]
preci_lat = pria['latitude'][:]
preci_lon = pria['longitude'][:]


path_data_inun = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/eventos_inundacion_sant2.xlsx'
data_inun = pd.read_excel(path_data_inun)

fec_inun = data_inun['Fecha']
fec_mun = data_inun['Municipio']
flat = data_inun['Latitud']
flon = data_inun['Longitud']



##fec = pres.variables['time']
##fec2 = fec[:]
##dat = np.array([num2date(y,units=fec.units) for y in fec2])

dates = pd.date_range('2018-01-01 00:00:00','2018-12-31 23:45:00',freq='1h')

########################################################################################################

rain_era = pd.DataFrame(pres.variables['tp'][:,0,0]*1000.,index=dates,columns=['ppt'])
rain_ime = pd.DataFrame(pris.variables['precipitationCal'][:,0,0],index=dates,columns=['ppt'])



for i,t in enumerate(fec_inun):

    t1 = t+ timedelta(hours=5)
    fd_evento = str(t1)[0:10]
    muni = fec_mun[i].replace(' ','_')
    #print(fd_evento[0:10])
    
    t2 = t + timedelta(hours=48)

    print(t1,t2)
    name = fd_evento+'_'+muni
    print(name)

    #rdates = dates[t1:t2]

    d1 = np.where(dates==t1)[0][0]
    d2 = np.where(dates==t2)[0][0]

    figg = fig = plt.figure(figsize=(10, 4))
    plt.title(name)
    plt.ylim(0,10)
    plt.plot(rain_ime[t1:t2],color='k',label='IMERG')
    plt.plot(rain_era[t1:t2],color='b',label='ERA5')
    plt.ylabel('Precipitacion [mm/h]')
    plt.legend()
    
    rsavee = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/EVENTOS/SERIES/'
    plt.savefig(rsavee+name,bbox_inches='tight',dpi=200)
    plt.close()


##    for j in range(d1,d2+1):
##
##        pre = preci[j,:,:].T
##
####        plt.imshow(pre)
####        plt.show()
####
####        asds
##        #print(pre.shape,preci_lat.shape,preci_lon.shape)
##
##
##        fig = plt.figure(figsize=(7, 7))
##        
##        plt.title(dates[j])
##
##        print(dates[j])
##
##        fh_evento = str(dates[j])
##        fh_evento = fh_evento[0:10]+'_'+fh_evento[11:].replace(':','_')
##        
##
##        m = Basemap(llcrnrlat=np.min(preci_lat),urcrnrlat=np.max(preci_lat),
##                llcrnrlon=np.min(preci_lon),urcrnrlon=np.max(preci_lon),
##                rsphere=6371200.,resolution='l',area_thresh=10000)
##
##        lons, lats = np.meshgrid(preci_lon, preci_lat)
##
##        x, y = m(lons, lats)
##
##        x1,y1 = m(flon, flat)
##        m.plot(x1[i], y1[i],'*',alpha=1,markersize=8,color='k')
##        
##        mapa = pre
##        cmap1 = 'pymeteo_radar'
##        
##
##        #print(mapa.shape,x.shape,y.shape)
##        bounds = np.arange(0.,9.1,0.1)
##        csf = m.contourf(x,y,mapa, cmap= cmap1,extend='max',levels=bounds)
##
##
##        m.drawparallels(np.arange(-90.,90.,0.5),labels=[1,0,0,0], size=8,linewidth=0.1,fmt='%2.2f')
##        m.drawmeridians(np.arange(0, 360, 0.5),labels=[0,1,0,1], size=8, linewidth=0.1,fmt='%2.2f')
##
##
##        #m.drawcoastlines(linewidth=0.8, color='k')
##        #m.drawmapboundary(linewidth=1, color='k')
##        m.drawcountries(linewidth=0.6, color='k')
##        
##        rutashape = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/SHAPE/santander/'
##        m.readshapefile(rutashape +'santander', 'Santander',
##                        linewidth=0.9, color='gray')
##
##
##        cax = plt.axes([0.88,0.15,0.018,0.685])
##        cbar = plt.colorbar(csf, cax=cax,orientation='vertical')
##        #cbar.ax.tick_params(labelsize=10)
##        cbar.set_label('Precipitacion [mm/h]',size=11)
##
##
##        to = 0.875
##        botto = 0.110
##        lef  = 0.115
##        righ = 0.840
##
##        hspac = 0.200
##        wspac = 0.200
##
##
##
##        fig.subplots_adjust(left=lef, bottom=botto, right=righ,
##                                top= to, wspace=wspac, hspace=hspac)
##
##        drsave = '/home/allyson/Documents/UNAL/Z_UIS/IA/PROYIA/PROY/IMG/EVENTOS/ESPACIAL/'
##
##
##        try:
##            os.stat(drsave+fd_evento+'_'+muni)
##        except:
##            os.mkdir(drsave+fd_evento+'_'+muni)
##
##        
##        
##        plt.savefig(drsave+fd_evento+'_'+muni+'/'+fh_evento,bbox_inches='tight',dpi=200)
##        plt.close()
##

    

















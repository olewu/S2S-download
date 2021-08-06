#------------import packages---------------#

import xarray as xr
import glob
import gridpp as gpp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import cmocean
import os
import subprocess
from scipy.spatial import cKDTree
import json

exec(open('/nird/projects/NS9001K/owul/projects/test/nearest_neighbour_kdtree.py').read())

#-------------set data paths----------#
#TODO: define paths somewhere else (e.g. a config file)
BW_path = '/nird/projects/NS9001K/owul/projects/S2S-download/data/BW/BW_temperature/'
S2SHC_path = '/nird/projects/NS9853K/DATA/S2S/hindcast/ECMWF/sfc/sst/'

#----------load a single (re-)forecast file for the grid---------#
S2SHC_files = glob.glob('{0:s}sst_*'.format(S2SHC_path))
print(S2SHC_files[0])

S2SHC_dset_test = xr.open_dataarray(S2SHC_files[0],engine='cfgrib',backend_kwargs={'indexpath':''})

#---------------put dimensions into a gridpp.Grid class----------------#
HC_lons,HC_lats = np.meshgrid(S2SHC_dset_test.longitude.values,S2SHC_dset_test.latitude.values)

# for land grid points set land_area_fractions to 0! (so they are not used as neighbours)
sea_gps = ~np.isnan(S2SHC_dset_test.data[0,0]) # 0 where land, 1 where sea
# if land_area_fraction is input, also altitudes must be passed to the class. set these to some constant value (0)

HC_lats_valid = HC_lats[sea_gps]

latlon_tree = KDTreeIndex(S2SHC_dset_test.sel(time="1999",step="1 days").squeeze(),valid=True)

#-----------import Barentswatch data-------------#
BW_files = glob.glob('{0:s}barentswatch_?????.nc'.format(BW_path))
print('{0:d} Barentswatch temperature files found'.format(len(BW_files)))

#-----------------get all stations with names and coordinates---------------#
stat_meta_file = ''
stat_metadat = []
with open(stat_meta_file) as f:
	stat_metadat.append(json.load(f))

for stat_id,BWf in enumerate(BW_files[:15]):
	BW_dset_test = xr.open_dataarray(BWf)
	#-----------put location into a gridpp.Point class----------#
	BW_lat,BW_lon = BW_dset_test.lat.values,BW_dset_test.lon.values
	site = BW_lat[0],BW_lon[0]
	print(site)
	
	#------------show the time span--------------#
	print('data from {:} until {:}'.format(BW_dset_test.time.min().values,BW_dset_test.time.max().values))
		
	d, inds, w, indslatlon = latlon_tree.query(site, k=10)

	print(HC_lats[indslatlon[0],indslatlon[1]],HC_lons[indslatlon[0],indslatlon[1]])

	#---------------plot map of sst (at one instance) and station to see how the nearest method works---------------#

	#-------------set a path for saving figures-------------#
	fpath = '/nird/projects/NS9001K/owul/projects/test/figures/'
	if not os.path.exists(fpath):
		subprocess.call('mkdir {0:s}'.format(fpath),shell=True)

	f,ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.AzimuthalEquidistant()})
	#ax = plt.axes(projection=ccrs.PlateCarree())
	ax.coastlines(resolution='10m')

	# slice the sst data:
	sst_slice = S2SHC_dset_test.loc["1999-09-30","1 days":"7 days",75:55,0:30].mean(dim='step')
	sst_slice.plot(ax=ax,transform=ccrs.PlateCarree())

	#levs = np.arange(280,290.1,.5)
	#bnrm = BoundaryNorm(levs,clip=True,ncolors=256)
	#h = ax.pcolormesh(HC_lons,HC_lats,S2SHC_dset_test.data[0,0],norm=bnrm,transform=ccrs.PlateCarree(),cmap=cmocean.cm.thermal)
	ax.plot(BW_lon,BW_lat,marker='x',color='r',transform=ccrs.PlateCarree())
	ax.plot(HC_lons[indslatlon[0],indslatlon[1]],HC_lats[indslatlon[0],indslatlon[1]],marker='+',color='r',transform=ccrs.PlateCarree())
	f.savefig('{0:s}test_nearest_gp_{1:s}.png'.format(fpath,str(stat_id)).zfill(3),bbox_inches='tight',dpi=300)
	plt.close(f)



#-------------------try a calibration of the closest grid point forecast for a single station--------------------#

#----------------Import packages---------------#
import xarray as xr
import glob
#import gridpp as gpp
# import numpy as np
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#from matplotlib.colors import BoundaryNorm
#import cmocean
# from datetime import datetime, timedelta
import os
import subprocess
# import json
import properscoring as ps

exec(open('/nird/projects/NS9001K/owul/projects/test/nearest_neighbour_kdtree.py').read())

#-----------------set paths-------------------------#
BW_path = '/nird/projects/NS9001K/owul/projects/S2S-download/data/BW/BW_temperature/'
S2SHC_path = '/nird/projects/NS9853K/DATA/S2S/hindcast/ECMWF/sfc/sst/'
fpath = '/nird/projects/NS9001K/owul/projects/test/figures/clbrtn/'
if not os.path.exists(fpath):
	subprocess.call('mkdir {0:s}'.format(fpath),shell=True)

#-----------------load data at one station---------------#
stat_id = 2 # INPUT the index (currently not written to retrieve by station id)

BW_files = glob.glob('{0:s}barentswatch_?????.nc'.format(BW_path))
BW_dset = xr.open_dataset(BW_files[stat_id])

fig,ax = plt.subplots(1,1)
BW_dset.sst.plot(ax=ax)
fig.savefig('{0:s}raw_BW_sst.png'.format(fpath),bbox_inches='tight',dpi=300)

site = BW_dset.lat.values[0],BW_dset.lon.values[0]

BW_sst = BW_dset.dropna('time')

#---------------get the full range of hindcasts-------------#
merge_file = 'sst_hc_2012-2019.nc'
TSLICE = slice('2011-11-30','2019-06-01')
if not os.path.exists(merge_file):
	sst_CY46_HC = sorted(glob.glob('{0:s}*CY46R1*cf*'.format(S2SHC_path)))

	HC_coll = []
	for ii,HC_file in enumerate(sst_CY46_HC):
		cf_dset = xr.open_dataset(HC_file,engine='cfgrib',backend_kwargs={'indexpath':''})
		pf_dset = xr.open_dataset(HC_file.replace('_cf_','_pf_'),engine='cfgrib',backend_kwargs={'indexpath':''})
		HC_coll.append(xr.concat([cf_dset.sortby('latitude',ascending=True).sel(time=TSLICE,latitude=slice(50,80),longitude=slice(0,40)),pf_dset.sortby('latitude',ascending=True).sel(time=TSLICE,latitude=slice(50,80),longitude=slice(0,40))],'number'))	

	HC_all = xr.concat(HC_coll,'time')
	HC_all.to_netcdf(merge_file)
else:
	HC_all = xr.open_dataset(merge_file)


test_arr = HC_all.isel(time=0).sel(step="1 days",number=0).squeeze().sst
latlon_tree = KDTreeIndex(test_arr,valid=True)
d, inds, w, indslatlon = latlon_tree.query(site, k=1)

HC_weekly_rm = HC_all.rolling(step=7,center=True).mean().sortby('time') - 273.15
HC_weekly_rm = HC_weekly_rm.rename({'number':'member'})

STEP = 13
fig,ax = plt.subplots(1,1)
BW_sst.sst.plot.line(ax=ax,x='time',color='C0',zorder=100,add_legend=False)
HC_sst_plot1 = HC_weekly_rm.sel(step='{0:d} days'.format(STEP)).isel(latitude=indslatlon[0],longitude=indslatlon[1]).squeeze().sst
HC_sst_plot1.plot.line(ax=ax,x='valid_time',color='lightgrey',zorder=0,add_legend=False)
HC_sst_plot1.mean('number').plot.line(ax=ax,x='valid_time',color='grey',zorder=1,add_legend=False)
fig.savefig('{0:s}HC_w3_sst.png'.format(fpath),bbox_inches='tight',dpi=300)

STEP = 20
fig,ax = plt.subplots(1,1)
BW_sst.sst.plot.line(ax=ax,x='time',color='C0',zorder=100,add_legend=False)
HC_sst_plot2 = HC_weekly_rm.sel(step='{0:d} days'.format(STEP)).isel(latitude=indslatlon[0],longitude=indslatlon[1]).squeeze().sst
HC_sst_plot2.plot.line(ax=ax,x='valid_time',color='lightgrey',zorder=0,add_legend=False)
HC_sst_plot2.mean('number').plot.line(ax=ax,x='valid_time',color='grey',zorder=1,add_legend=False)
fig.savefig('{0:s}HC_w4_sst.png'.format(fpath),bbox_inches='tight',dpi=300)


#----------Find matching dates---------#
t_ind = []
for tt_BW in BW_sst.sel(time=TSLICE).time.values:
	t_ind.append(abs(HC_sst_plot2.valid_time - tt_BW).argmin())
t_ind = xr.concat(t_ind,dim='valid_time')

fig,ax = plt.subplots(1,1)
ax.scatter(HC_sst_plot2.isel(time=t_ind).mean('member'),BW_sst.sel(time=TSLICE).sst)
fig.savefig('{0:s}HC_BW_scatter.png'.format(fpath),bbox_inches='tight',dpi=300)

# the above arrays in the scatter plot have the same size (269) now
#---------skill of raw forecast---------#
CRPS_raw = ps.crps_ensemble(BW_sst.sel(time=TSLICE).sst.squeeze().values,HC_sst_plot2.isel(time=t_ind).values).mean()
print(CRPS_raw)

#------------Calibrate using NGR--------------#
# define the optimization function:


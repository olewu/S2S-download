import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

class KDTreeIndex():    
    """ A KD-tree implementation for fast point lookup on a 2D grid
    
    Keyword arguments: 
    dataset -- a xarray DataArray containing lat/lon coordinates
               (named 'lat' and 'lon' respectively)
    

	Copied (2021-06-18) from https://www.guillaumedueymes.com/post/ckdtree_netcdf/
	adapted to include treating of masked points on grid             
    """    
    def transform_coordinates(self, coords):
        """ Transform coordinates from geodetic to cartesian
        
        Keyword arguments:
        coords - a set of lan/lon coordinates (e.g. a tuple or 
                 an array of tuples)
        """
        # WGS 84 reference coordinate system parameters
        A = 6378.137 # major axis [km]   
        E2 = 6.69437999014e-3 # eccentricity squared    
        
        coords = np.asarray(coords).astype(float)
        
        # is coords a tuple? Convert it to an one-element array of tuples
        if coords.ndim == 1:
            coords = np.array([coords])
        
        # convert to radiants
        lat_rad = np.radians(coords[:,0])
        lon_rad = np.radians(coords[:,1]) 
        
        # convert to cartesian coordinates
        r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
        x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
        z = r_n * (1 - E2) * np.sin(lat_rad)
        
        return np.column_stack((x, y, z))
    
    def __init__(self, dataset, valid=False):
        # store original dataset shape
        self.shape = dataset.shape
        self.valid = valid
 
        try:
            lon2d, lat2d = np.meshgrid(dataset.lon, dataset.lat)
        except:
            lon2d, lat2d = np.meshgrid(dataset.longitude,dataset.latitude)
        # reshape and stack coordinates
        coords = np.column_stack((lat2d.ravel(),
                                  lon2d.ravel()))

        # get the valid coordinates in case of data array with nans (masked)
        # find which of the dimensions are lat and lon:
        lalodim_idx = np.logical_or(np.array([*dataset.shape]) == dataset.latitude.shape[0],np.array([*dataset.shape]) == dataset.longitude.shape[0])
        
        idx_valid = ~np.isnan(dataset.values).ravel()
        self.all_index = np.arange(len(coords))[idx_valid]
        coords_valid = np.column_stack((lat2d.ravel()[idx_valid],
                                        lon2d.ravel()[idx_valid]))

        # construct KD-tree
        if valid:
            self.tree = cKDTree(self.transform_coordinates(coords_valid))
        else:
            self.tree = cKDTree(self.transform_coordinates(coords))
        
    def query(self, point, k):
        """ Query the kd-tree for nearest neighbour.
        Keyword arguments:
        point -- a (lat, lon) tuple or array of tuples
        """
        d, inds  = self.tree.query(self.transform_coordinates(point),k=k)
        w = 1.0 / d**2
        # regrid to 2D grid
        if self.valid:
            indslatlon = np.unravel_index(self.all_index[inds],self.shape)
        else:
            indslatlon = np.unravel_index(inds, self.shape)
        return d, inds, w, indslatlon
    
    def query_ball_point(self, point, radius):
        """ Query the kd-tree for all point within distance 
        radius of point(s) x
        
        Keyword arguments:
        point -- a (lat, lon) tuple or array of tuples
        radius -- the search radius (km)
        """
        
        index = self.tree.query_ball_point(self.transform_coordinates(point),
                                           radius)

        # regrid to 2D grid
        if self.valid:
            index = np.unravel_index(self.all_index[index[0]],self.shape)
        else:
            index = np.unravel_index(index[0], self.shape)
        
        # return DataArray indexers
        return xr.DataArray(index[0], dims='lat'), \
               xr.DataArray(index[1], dims='lon')

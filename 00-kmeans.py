#!/usr/bin/env python
# coding: utf-8

# # Land cover classification at the Mississppi Delta
# 
# In this notebook, you will use a k-means **unsupervised** clustering algorithm to group pixels by similar spectral signatures. k-means is an **exploratory** method for finding patterns in data. Because it is unsupervised, you don't need any training data for the model. You also can't tell how well it "performs" because the clusters will not correspond to any particular land cover class. However, we expect at least some of the clusters to be identifiable.
# 
# You will use the [harmonized Sentinal/Landsat multispectral dataset](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf). You can access the data with an [Earthdata account](https://www.earthdata.nasa.gov/learn/get-started) by installing the [`earthaccess` library from NSIDC](https://github.com/nsidc/earthaccess):
# 
# ```bash
# conda install -c conda-forge -y earthaccess
# ```

# ## STEP 1: SET UP

# YOUR TASK:
#   1. Import all libraries you will need for this analysis
#   2. Configure GDAL parameters to help avoid connection errors:
#      ```python
#      os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
#      os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"
#      ```

# In[1]:


# BEGIN SOLUTION
import os
import pickle
import re
import warnings

import cartopy.crs as ccrs
import earthaccess
import earthpy as et
import geopandas as gpd
import geoviews as gv
import hvplot.pandas
import hvplot.xarray
import numpy as np
import pandas as pd
import rioxarray as rxr
import rioxarray.merge as rxrmerge
import xarray as xr
from ipywidgets import IntProgress
from IPython.display import display
from shapely.geometry import Polygon
from sklearn.cluster import KMeans

os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

HUC_LEVEL = 12

warnings.simplefilter('ignore')
# END SOLUTION


# Below you can find code for a caching **decorator** which you can use in your code. To use the decorator:
# 
# ```python
# @cached(key, override)
# def do_something(*args, **kwargs):
#     ...
#     return item_to_cache
# ```
# 
# This decorator will **pickle** the results of running the `do_something()` function, and only run the code if the results don't already exist. To override the caching, for example temporarily after making changes to your code, set `override=True`.

# In[2]:


def cached(key, override=False):
    """
    A decorator to cache function results
    
    Parameters
    ==========
    key: str
      File basename used to save pickled results
    override: bool
      When True, re-compute even if the results are already stored
    """
    def compute_and_cache_decorator(compute_function):
        """
        Wrap the caching function
        
        Parameters
        ==========
        compute_function: function
          The function to run and cache results
        """
        def compute_and_cache(*args, **kwargs):
            """
            Perform a computation and cache, or load cached result.
            
            Parameters
            ==========
            args
              Positional arguments for the compute function
            kwargs
              Keyword arguments for the compute function
            """
            filename = os.path.join(
                et.io.HOME, et.io.DATA_NAME, 'jars', f'{key}.pickle')
            
            # Check if the cache exists already or override caching
            if not os.path.exists(filename) or override:
                # Make jars directory if needed
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Run the compute function as the user did
                result = compute_function(*args, **kwargs)
                
                # Pickle the object
                with open(filename, 'wb') as file:
                    pickle.dump(result, file)
            else:
                # Unpickle the object
                with open(filename, 'rb') as file:
                    result = pickle.load(file)
                    
            return result
        
        return compute_and_cache
    
    return compute_and_cache_decorator


# ## STEP 2: STUDY SITE

# For this analysis, you will use a watershed from the [Water Boundary Dataset](https://www.usgs.gov/national-hydrography/access-national-hydrography-products), HU12 watersheds (WBDHU12.shp).
# 
# YOUR TASK:
#   1. Download the Water Boundary Dataset for region 8 (Mississippi)
#   2. Select watershed 080902030506
#   3. Generate a site map of the watershed

# In[3]:


# BEGIN SOLUTION
@cached(f'wbd_08_hu{HUC_LEVEL}_gdf')
def read_wbd_file(wbd_filename):
    # Download and unzip
    wbd_url = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/"
        f"{wbd_filename}.zip")
    wbd_dir = et.data.get_data(url=wbd_url)
                  
    # Read desired data
    wbd_path = os.path.join(wbd_dir, 'Shape', f'WBDHU{HUC_LEVEL}.shp')
    wbd_gdf = gpd.read_file(wbd_path, engine='pyogrio')
    return wbd_gdf

wbd_gdf = read_wbd_file("WBD_08_HU2_Shape")

delta_gdf = wbd_gdf[wbd_gdf[f'huc{HUC_LEVEL}'].isin(['080902030506'])].dissolve()
(
    delta_gdf.to_crs(ccrs.Mercator())
    .hvplot(alpha=.2, fill_color='white', tiles='EsriImagery', crs=ccrs.Mercator())
    .opts(width=600, height=300)
)
# END SOLUTION


# I chose this watershed because it covers parts of New Orleans an is near the Mississippi Delta. Deltas are boundary areas between the land and the ocean, and as a result tend to contain a rich variety of different land cover and land use types.
# 
# In the cell below, write a 2-3 sentence **site description** of this area that helps to put your analysis in context.

# **YOUR SITE DESCRIPTION HERE**
# 
# ![](https://media.baamboozle.com/uploads/images/510741/1651543763_75056_gif-url.gif)

# ## STEP 3: MULTISPECTRAL DATA

# ### Search for data
# 
# **YOUR TASK:**
#   1. Log in to the `earthaccess` service using your Earthdata credentials:
#      ```python
#      earthaccess.login(persist=True)
#      ```
#   2. Modify the following sample code to search for granules of the HLSL30 product overlapping the watershed boundary from May to October 2023 (there should be 76 granules):
#      ```python
#      results = earthaccess.search_data(
#          short_name="...",
#          cloud_hosted=True,
#          bounding_box=tuple(gdf.total_bounds),
#          temporal=("...", "..."),
#      )
#      ```

# In[11]:


# BEGIN SOLUTION
earthaccess.login(persist=True)
results = earthaccess.search_data(
    short_name="HLSL30",
    cloud_hosted=True,
    bounding_box=tuple(delta_gdf.total_bounds),
    temporal=("2023-05", "2023-10"),
)
# END SOLUTION


# ### Compile information about each granule
# 
# I recommend building a GeoDataFrame, as this will allow you to plot the granules you are downloading and make sure they line up with your shapefile. You could also use a DataFrame, dictionary, or a custom object to store this information.
# 
# **YOUR TASK -- Write a function that:**
#   1. For each search result:
#       1. Get the following information (HINT: look at the ['umm'] values for each search result):
#           - granule id (UR)
#           - datetime
#           - geometry (HINT: check out the shapely.geometry.Polygon class to convert points to a Polygon)
#       2. Open the granule files. I recomment opening one granule at a time, e.g. with (`earthaccess.open([result]`).
#       3. For each file (band), get the following information:
#           - file handler returned from `earthaccess.open()`
#           - tile id
#           - band number
#   2. Compile all the information you collected into a GeoDataFrame

# In[5]:


# BEGIN SOLUTION
def get_earthaccess_links(results):
    f = IntProgress(min=0, max=len(results), description='Open granules:')
    display(f)
    
    url_re = re.compile(r'\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif')

    link_rows = []
    url_dfs = []
    for granule in results:
        # Get granule information
        info_dict = granule['umm']
        granule_id = info_dict['GranuleUR']
        datetime = pd.to_datetime(
            info_dict
            ['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        points = (
            info_dict
            ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
            ['Boundary']['Points'])
        geometry = Polygon([(point['Longitude'], point['Latitude']) for point in points])
        
        # Get URL
        files = earthaccess.open([granule])
        for file in files:
            match = url_re.search(file.full_name)
            if match is not None:
                link_rows.append(
                    gpd.GeoDataFrame(
                        dict(
                            datetime=[datetime],
                            tile_id=[match.group('tile_id')],
                            band=[match.group('band')],
                            url=[file],
                            geometry=[geometry]
                        ),
                        crs="EPSG:4326"
                    )
                )
        # Increment progress bar
        f.value += 1
    file_df = pd.concat(link_rows).reset_index(drop=True)
    return file_df
# END SOLUTION


# ### Open, crop, and mask data
# 
# This will be the most resource-intensive step. I recommend caching your results using the `cached` decorator or by writing your own caching code. I also recommend testing this step with one or two dates before running the full computation.
# 
# This code should include at least one **function** including a numpy-style docstring. A good place to start would be a function for opening a single masked raster, applying the appropriate scale parameter, and cropping.
# 
# **YOUR TASK:**
# 1. For each granule:
#     1. Open the Fmask band, crop, and compute a quality mask for the granule. You can use the following code as a starting point, making sure that `mask_bits` contains the quality bits you want to consider:
#          ```python
#          # Expand into a new dimension of binary bits
#          bits = np.unpackbits(da.astype(np.uint8)).reshape(da.shape + (-1,))
# 
#          # Select the required bits and check if any are flagged
#          mask = np.prod(bits[..., mask_bits]==0, axis=-1)
#          ```
# 
#     2. For each band that starts with 'B':
#         1. Open the band, crop, and apply the scale factor
#         2. Name the DataArray after the band using the `.name` attribute
#         3. Apply the cloud mask using the `.where()` method
#         4. Store the DataArray in your data structure (e.g. adding a GeoDataFrame column with the DataArray in it. Note that you will need to remove the rows for unused bands)

# In[7]:


# BEGIN SOLUTION
@cached('delta_reflectance_da_df')
def compute_reflectance_da(search_results, boundary_gdf):
    """
    Connect to files over VSI, crop, cloud mask, and wrangle
    
    Returns a single reflectance DataFrame 
    with all bands as columns and
    centroid coordinates and datetime as the index.
    
    Parameters
    ==========
    file_df : pd.DataFrame
        File connection and metadata (datetime, tile_id, band, and url)
    boundary_gdf : gpd.GeoDataFrame
        Boundary use to crop the data
    """
    def open_dataarray(url, boundary_proj_gdf, scale=1, masked=True):
        # Open masked DataArray
        da = rxr.open_rasterio(url, masked=masked).squeeze() * scale
        
        # Reproject boundary if needed
        if boundary_proj_gdf is None:
            boundary_proj_gdf = boundary_gdf.to_crs(da.rio.crs)
            
        # Crop
        cropped = da.rio.clip_box(*boundary_proj_gdf.total_bounds)
        return cropped
    
    def compute_quality_mask(da, mask_bits=[1, 2, 3]):
        """Mask out low quality data by bit"""
        # Unpack bits into a new axis
        bits = (
            np.unpackbits(
                da.astype(np.uint8), bitorder='little'
            ).reshape(da.shape + (-1,))
        )

        # Select the required bits and check if any are flagged
        mask = np.prod(bits[..., mask_bits]==0, axis=-1)
        return mask

    file_df = get_earthaccess_links(search_results)
    
    f = IntProgress(min=0, max=len(file_df), description='Open granules:')
    display(f)
    
    granule_da_rows= []
    boundary_proj_gdf = None

    for (datetime, tile_id), granule_df in file_df.groupby(['datetime', 'tile_id']):
        print(f'Processing granule {tile_id} {datetime}')
              
        # Open granule cloud cover
        cloud_mask_url = granule_df.loc[granule_df.band=='Fmask', 'url'].values[0]
        cloud_mask_cropped_da = open_dataarray(cloud_mask_url, boundary_proj_gdf, masked=False)

        # Compute cloud mask
        cloud_mask = compute_quality_mask(cloud_mask_cropped_da)

        da_list = []
        df_list = []
        for i, row in granule_df.iterrows():
            
            if row.band.startswith('B'):
                band_cropped = open_dataarray(
                    row.url, boundary_proj_gdf, scale=0.0001)
                band_cropped.name = row.band
                row['da'] = band_cropped.where(cloud_mask)
                granule_da_rows.append(row.to_frame().T)

            # Increment progress bar
            f.value += 1
    
    return pd.concat(granule_da_rows)

reflectance_da_df = compute_reflectance_da(results, delta_gdf)
# END SOLUTION


# ### Merge and Composite Data
# 
# You will notice for this watershed that:
# 1. The raster data for each date are spread across 4 granules
# 2. Any given image is incomplete because of clouds
# 
# **YOUR TASK:**
# 1. For each band:
#     1. For each date:
#         1. Merge all 4 granules
#         2. Mask any negative values created by interpolating from the nodata value of -9999 (`rioxarray` should account for this, but doesn't appear to when merging. If you leave these values in they will create problems down the line)
#     2. Concatenate the merged DataArrays along a new date dimension
#     3. Take the mean in the date dimension to create a composite image that fills cloud gaps
#     4. Add the band as a dimension, and give the DataArray a name
# 2. Concatenate along the band dimension

# In[8]:


@cached('delta_reflectance_da')
def merge_and_composite_arrays(granule_da_df):
    f = IntProgress(min=0, max=len(granule_da_df)/4, description='Merge arrays:')
    display(f)
    
    # Merge and composite and image for each band
    df_list = []
    da_list = []
    for band, band_df in granule_da_df.groupby('band'):
        merged_das = []
        for datetime, date_df in band_df.groupby('datetime'):
            # Merge granules for each date
            merged_da = rxrmerge.merge_arrays(list(date_df.da))
            # Mask negative values
            merged_da = merged_da.where(merged_da>0)
            merged_das.append(merged_da)
            # Increment progress bar
            f.value += 1
            
        # Composite images across dates
        composite_da = xr.concat(merged_das, dim='datetime').median('datetime')
        composite_da['band'] = int(band[1:])
        composite_da.name = 'reflectance'
        da_list.append(composite_da)
        
    return xr.concat(da_list, dim='band')

reflectance_da = merge_and_composite_arrays(reflectance_da_df)
reflectance_da


# ## STEP 4: K-MEANS

# Cluster your data by spectral signature using the k-means algorithm. 
# 
# **YOUR TASK:**
# 1. Convert your DataArray into a **tidy** DataFrame of reflectance values (hint: check out the `.unstack()` method)
# 2. Filter out all rows with no data (all 0s or any N/A values)
# 3. Fit a k-means model. You can experiment with the number of groups to find what works best.

# In[9]:


# BEGIN SOLUTION
model_df = reflectance_da.to_dataframe().reflectance.unstack('band')
model_df = model_df.drop(columns=[10, 11]).dropna()

# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction = KMeans(n_clusters=6).fit_predict(model_df.values)

# Reshaping the prediction array to match the original raster shape
model_df['clusters'] = prediction
model_df
# END SOLUTION


# In[13]:





# ## STEP 5: PLOT

# **YOUR TASK:**
# Create a plot that shows the k-means clusters next to an RGB image of the area. You may need to brighten your RGB image by multiplying it by 10.

# In[10]:


# BEGIN SOLUTION
rgb = reflectance_da.sel(band=[4, 3, 2])
rgb_uint8 = (rgb * 255).astype(np.uint8).where(rgb!=np.nan)
rgb_bright = rgb_uint8 * 10
rgb_sat = rgb_bright.where(rgb_bright < 255, 255)

(
    rgb_sat.hvplot.rgb( 
        x='x', y='y', bands='band',
        data_aspect=1,
        xaxis=None, yaxis=None)
    + model_df.clusters.to_xarray().sortby(['x', 'y']).hvplot(cmap="Colorblind", aspect='equal') 
)
# END SOLUTION


# Don't forget to interpret your plot!

# **YOUR PLOT HEADLINE AND DESCRIPTION HERE**
# 
# ![](https://media.baamboozle.com/uploads/images/510741/1651543763_75056_gif-url.gif)

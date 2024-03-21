#!/usr/bin/env python
# coding: utf-8

# <div><img style="float: left; padding-right: 3em;" src="https://avatars.githubusercontent.com/u/19476722" width="150" /><div/>
# 
# # Earth Data Science Coding Challenge!
# Before we get started, make sure to read or review the guidelines below. These will help make sure that your code is **readable** and **reproducible**. 

# ## Don't get **caught** by these Jupyter notebook gotchas
# 
# <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*o0HleR7BSe8W-pTnmucqHA.jpeg" width=300 style="padding: 1em; border-style: solid; border-color: grey;" />
# 
#   > *Image source: https://alaskausfws.medium.com/whats-big-and-brown-and-loves-salmon-e1803579ee36*
# 
# These are the most common issues that will keep you from getting started and delay your code review:
# 
# 1. When you try to run some code on GitHub Codespaces, you may be prompted to select a **kernel**.
#    * The **kernel** refers to the version of Python you are using
#    * You should use the **base** kernel, which should be the default option. 
#    * You can also use the `Select Kernel` menu in the upper right to select the **base** kernel
# 2. Before you commit your work, make sure it runs **reproducibly** by clicking:
#    1. `Restart` (this button won't appear until you've run some code), then
#    2. `Run All`
# 
# ## Check your code to make sure it's clean and easy to read
# 
# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSO1w9WrbwbuMLN14IezH-iq2HEGwO3JDvmo5Y_hQIy7k-Xo2gZH-mP2GUIG6RFWL04X1k&usqp=CAU" height=200 />
# 
# * Format all cells prior to submitting (right click on your code).
# * Use expressive names for variables so you or the reader knows what they are. 
# * Use comments to explain your code -- e.g. 
#   ```python
#   # This is a comment, it starts with a hash sign
#   ```
# 
# ## Label and describe your plots
# 
# ![Source: https://xkcd.com/833](https://imgs.xkcd.com/comics/convincing.png)
# 
# Make sure each plot has:
#   * A title that explains where and when the data are from
#   * x- and y- axis labels with **units** where appropriate
#   * A legend where appropriate
# 
# 
# ## Icons: how to use this notebook
# We use the following icons to let you know when you need to change something to complete the challenge:
#   * &#128187; means you need to write or edit some code.
#   
#   * &#128214;  indicates recommended reading
#   
#   * &#9998; marks written responses to questions
#   
#   * &#127798; is an optional extra challenge
#   

# ---

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

# In[2]:


import os
import pathlib
import pickle

import earthaccess
import earthpy as et
import earthpy.earthexplorer as etee
import geopandas as gpd
import geoviews as gv
import gitpass
import glob
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
import rioxarray.merge as rxrm
import shapely
import xarray as xr
import xrspatial
from sklearn.cluster import KMeans

data_dir = os.path.join(et.io.HOME, et.io.DATA_NAME)

os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"


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

# In[4]:


# Caching decorator

def cached(key, override=False):
    """"""
    def compute_and_cache_decorator(compute_function):
        """"""
        def compute_and_cache(*args, **kwargs):
            """Perform a computation and cache, or load cached result"""
            filename = os.path.join(et.io.HOME, et.io.DATA_NAME, 'jars', f'{key}.pickle')
            
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

# In[4]:


#Download WBD spatial data

wbd_path = os.path.join(
    data_dir,
    'earthpy-downloads',
    'WBD_08_HU2_Shape',
    'Shape',
    'WBDHU12.shp'
)

wbd_url = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/"
           "Hydrography/WBD/HU2/Shape/WBD_08_HU2_Shape.zip"
          )

if not os.path.exists(wbd_path):
    print('downloading ' + wbd_url)  
    wbd_zip = et.data.get_data(url=wbd_url)
else:
    print(wbd_path + " already exists")

# Create a GDF and plot the HUC    
    
wbd_gdf = gpd.read_file(wbd_path)
huc_gdf = wbd_gdf[wbd_gdf['huc12'] == '080902030506']
gv.tile_sources.EsriImagery() * gv.Polygons(huc_gdf).opts(
    fill_alpha=0, line_color='blue', title='HUC 080902030506',
    height=600, width=600
)


# I chose this watershed because it covers parts of New Orleans an is near the Mississippi Delta. Deltas are boundary areas between the land and the ocean, and tend to contain a rich variety of different land cover and land use types.
# 
# In the cell below, write a 2-3 sentence **site description** of this area that helps to put your analysis in context.

# **Site Description**
# 
# HUC 080902030506 is a HUC 12 watershed located in the Mississippi Delta southeast of New Orleans. HUC 12 watersheds are the smallest geographical unit within the Watershed Boundary Dataset and are often analyzed to assess watersheds on a local level. This watershed consists of wetlands, open water, floodplains, upland islands, and some manmade features such as levees.

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

# In[5]:


# Search multispectral data from Earthdata

earthaccess.login(persist=True)

results = earthaccess.search_data(
    short_name="HLSL30",
    cloud_hosted=True,
    bounding_box=tuple(huc_gdf.total_bounds),
    temporal=("2023-05-01", "2023-09-30"),
)


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


# Open the Earthaccess results so you only have to do it once
@cached("open_results", False)
def open_results_and_cache(results):
    open_results = earthaccess.open(results)
    return open_results
open_results = open_results_and_cache(results)


# In[7]:


# Process the granules from Earthdata
def create_granule_df(earthdata_results, open_results):
    """
    Process the granules returned by an Earthdata search

    Parameters
    ==========
    earthdata_results : list
      A list object returned by earthacces.search_data().
      This contains the result granules from the Earthdata search.
      
    open_results : opened earthdata_results object

    Returns
    =======
    granule_df : DataFrame
      A dataframe with a row for each granule file.
      Each row will contain the granule ID, the download URL,
      the datetime, the tile ID, and the band.
    """
    columns = ['geometry','granule_id', 'file_url', 'tile_id', 'datetime', 'band_number']
    granule_df = gpd.GeoDataFrame(columns=columns)
    # Loop through each file in the granule results and accumulate attributes
    for url in open_results:
        file_url = str(url.url)
        # Should probably use regular expression here
        gran_id = file_url[73:107]
        tile_id = file_url[81:87]
        datetime = file_url[88:95]
        band = file_url[-7:-4]
        if band == 'ask':
            band = 'Fmask'
        # Find granule geometry based on granule ID
        for result in earthdata_results:
            result_attr = result['umm']
            result_gran_id = result_attr['GranuleUR']
            if result_gran_id == gran_id:
                extent = result_attr['SpatialExtent']
                geom_extent=extent['HorizontalSpatialDomain']['Geometry']['GPolygons']
                coord_list = []
                for item in geom_extent:
                    poly = item['Boundary']['Points']
                    for vertex in poly:
                        longitude = vertex['Longitude']
                        latitude = vertex['Latitude']
                        coord_list.append((longitude, latitude))
                granule_geometry = shapely.geometry.Polygon(coord_list)
        # Append info for each file to dataframe
        new_row = pd.DataFrame({'geometry': [granule_geometry],
                   'granule_id': [gran_id],
                   'file_url': [file_url],
                   'tile_id': [tile_id],
                   'datetime': [datetime],
                   'band_number': [band]
                  })
        granule_df = pd.concat([granule_df, new_row], ignore_index=True)
    return granule_df


# In[8]:


@cached("download_df", False)
def do_something(results, open_results):
    download_df = create_granule_df(results, open_results)
    return download_df
download_df = do_something(results, open_results)


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

# In[9]:


# Function to compute image mask
def process_tifs(download_df, huc_gdf):
    """
    Load and process the Fmask and B band .tif files
    in a granule.

    Parameters
    ==========
    download_df : pandas.core.groupby.generic.DataFrame
      Dataframe containing all the rows for each band .tif for all the files.
    
    huc_gdf : GeoDataFrame
      Geodataframe containing the geometry of the HUC watershed used to clip

    Returns
    =======
    accumulator_df : Pandas dataframe
      Dataframe containing the processed rows for all the tifs.
      An xarray has been added for each processed .tif in the last column.
    """
    grouped = download_df.groupby('granule_id')
    columns = ['geometry',
               'granule_id', 
               'file_url', 
               'tile_id', 
               'datetime', 
               'band_number',
               'processed_da']
    accumulator_df = pd.DataFrame(columns=columns)
    # Looping through each Granule and processing data
    # Create cloud mask
    for granule_id, group_data in grouped:
        print("Processing data for Granule ID:", granule_id)
        granule_gdf = group_data
        fmask_gdf = group_data['band_number'] == "Fmask"
        fmask_gdf = group_data[fmask_gdf].iloc[0:1]
        file_name = (
            fmask_gdf.iloc[0]['granule_id'] +
            "." + fmask_gdf.iloc[0]['band_number'] + '.tif'
        )
        file_path = data_dir + '\\earthpy-downloads\\' + file_name
        fmask_da = rxr.open_rasterio(file_path).squeeze()
        huc_gdf_crs = huc_gdf.to_crs(fmask_da.rio.crs)
        fmask_crop_da = fmask_da.rio.clip_box(*huc_gdf_crs.total_bounds)
        mask = compute_mask(fmask_crop_da)
        # Apply crop, scale factor, and mask to all "B" files
        for index, row in group_data.iterrows():
            if (row['band_number'].startswith('B')
                and row['band_number'] not in ('B10', 'B11')):
                file_name = row['granule_id'] + "." + row['band_number'] + '.tif'
                file_id = row['granule_id'] + "." + row['band_number']
                file_path = data_dir + '\\earthpy-downloads\\' + file_name
                bband_da = rxr.open_rasterio(file_path).squeeze()
                huc_gdf = huc_gdf.to_crs(bband_da.rio.crs)
                bband_crop_da = bband_da.rio.clip_box(*huc_gdf_crs.total_bounds)
                bband_crop_da = bband_crop_da.where(bband_crop_da >= 0, np.nan)
                scale_factor = 0.0001 
                bband_crop_da = bband_crop_da * scale_factor
                processed_da = bband_crop_da.where(mask)
                add_row = pd.DataFrame({'geometry': row['geometry'],
                    'granule_id': row['granule_id'],
                    'file_url': row['file_url'],
                    'tile_id': row['tile_id'],
                    'datetime': row['datetime'],
                    'band_number': row['band_number'],
                    'processed_da': [processed_da]})
                accumulator_df = pd.concat([accumulator_df, add_row])
    return accumulator_df

def compute_mask(da, mask_bits = [1,2,3]):
    """
    Compute a mask layer from the Fmask

    Parameters
    ==========
    da : rioxarray
      An array of the Fmask layer to use to compute the mask
      
    bits : list
      A list of bits to exclude, documentation is here on page 21:
      https://hls.gsfc.nasa.gov/wp-content/uploads/2019/01/HLS.v1.4.UserGuide_draft_ver3.1.pdf

    Returns
    =======
    mask : rioxarray
      An array with computed mask values
    """
    # Unpack bits in the Fmask array
    # Need to reverse order of bits from most significant to little
    bits = np.unpackbits(da.astype(np.uint8), bitorder = 'little').reshape(da.shape + (-1,))
    
    # Select the bits to use for the mask
    mask = np.prod(bits[..., mask_bits]==0, axis=-1)
    return mask
    


# In[10]:


# Run methods to process all .tifs and create accumulator dataframe
@cached("accumulator_df", False)
def process_tifs_and_cache(download_df):
    accumulator_df = process_tifs(download_df, huc_gdf)
    return accumulator_df
accumulator_df = process_tifs_and_cache(download_df)


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

# In[11]:


# Group the DataFrame by 'band_number' and 'datetime'
grouped_band_df = accumulator_df.groupby('band_number')

# Iterate through each band
band_accumulator = []
for band_number, band_data in grouped_band_df:
    # Iterate over each date
    date_accumulator = []
    for datetime, date_data in band_data.groupby('datetime'):
        # Merge data from all 4 granules
        merged_date_array = rxrm.merge_arrays(date_data['processed_da'].values)
        # Add datetime dimension
        merged_date_array = merged_date_array.assign_coords(date=datetime)
        # Mask any negative values
        merged_date_array = xr.where(merged_date_array <= 0, np.nan, merged_date_array)
        # Append to accumulator list
        date_accumulator.append(merged_date_array)
    
    # Concatenate the merged DataArrays along a new 'date' dimension
    date_accumulator = xr.concat(date_accumulator, dim='datetime')
    # Take the mean along the 'date' dimension to create a composite image
    composite_image = date_accumulator.mean(dim='datetime', skipna=True)
    
    # Add the 'band_number' as a new dimension and give the DataArray a name
    composite_image = composite_image.assign_coords(band_number=band_number)
    #composite_image.name = str(band_number) + "_array"
    #print(composite_image.name)
    
    band_accumulator.append(composite_image)

# Concatenate along the 'band_number' dimension
all_bands_array = xr.concat(band_accumulator, 'band_number')



# ## STEP 4: K-MEANS

# Cluster your data by spectral signature using the k-means algorithm. 
# 
# **YOUR TASK:**
# 1. Convert your DataArray into a **tidy** DataFrame of reflectance values (hint: check out the `.unstack()` method)
# 2. Filter out all rows with no data (all 0s or any N/A values)
# 3. Fit a k-means model. You can experiment with the number of groups to find what works best.

# In[14]:


# Organize data array of bands
all_bands_array.name = 'reflectance'
model_df = (
    all_bands_array
    .sel(band_number=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B09'])
    .to_dataframe()
    .reflectance
    .unstack('band_number')
    .dropna()
)

# Fit KMeans model to my data
k_means = KMeans(n_clusters=6)
model_df['category'] = k_means.fit_predict(model_df)


# ## STEP 5: PLOT

# **YOUR TASK:**
# Create a plot that shows the k-means clusters next to an RGB image of the area. You may need to brighted your RGB image by multiplying it by 10.

# In[23]:


# Prepare data to display KMeans categories
model_array = model_df.category.to_xarray()
model_plot = (model_df
              .category
              .to_xarray()
              .sortby(['x','y'])
              .hvplot(x='x', y='y', colormap='Colorblind', aspect=1, title='KMeans Categories')
             )

# Prepare the data to display as RGB
rgb = all_bands_array.sel(band_number=['B04', 'B03', 'B02'])
rgb_unint8 = (rgb * 255).astype(np.uint8())
rgb_brighten = rgb_unint8 * 10
rgb_plot = (
    rgb_brighten
    .hvplot
    .rgb(y='y', x='x', bands='band_number', aspect=1, colormap='RGB', title='RGB Aerial Image')
)

rgb_plot + model_plot


# Don't forget to interpret your plot!

# **Unsupervised Land Cover Classification for HUC 080902030506**
# 
# Visual inspection of the RGB image lets us assume that this watershed is dominated by land cover classes including open water, wetlands, wetland transition zones, a variety of upland vegetated communities, and developed areas such as roads and levees.
# 
# We ran K Means classification using 6 classification categories. Inspecting the results of the classification along side the RGB image shows that K Means does a fairly good job of classifying major land cover types specifically open water.
# 
# It appears that upland categories are being classified as well but it is difficult to assess the accuracy of these categories without having on the ground validation of what different communities exist in the upland areas. As such, K Means is a useful exploratory tool but we can not use it to confidently identify and classify areas of land cover dominated by diverse vegetation and wetland types.

# In[ ]:





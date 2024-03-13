# %% [markdown]
# <div><img style="float: left; padding-right: 3em;" src="https://avatars.githubusercontent.com/u/19476722" width="150" /><div/>
# 
# # Earth Data Science Coding Challenge!
# Before we get started, make sure to read or review the guidelines below. These will help make sure that your code is **readable** and **reproducible**. 

# %% [markdown]
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

# %% [markdown]
# ---

# %% [markdown]
# # Land cover classification at the Mississppi Delta
# 
# In this notebook, you will use a k-means **unsupervised** clustering algorithm to group pixels by similar spectral signatures. k-means is an **exploratory** method for finding patterns in data. Because it is unsupervised, you don't need any training data for the model. You also can't tell how well it "performs" because the clusters will not correspond to any particular land cover class. However, we expect at least some of the clusters to be identifiable.
# 
# You will use the [harmonized Sentinal/Landsat multispectral dataset](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf). You can access the data with an [Earthdata account](https://www.earthdata.nasa.gov/learn/get-started) by installing the [`earthaccess` library from NSIDC](https://github.com/nsidc/earthaccess):
# 
# ```bash
# conda install -c conda-forge -y earthaccess
# ```

# %% [markdown]
# ## STEP 1: SET UP

# %% [markdown]
# YOUR TASK:
#   1. Import all libraries you will need for this analysis
#   2. Configure GDAL parameters to help avoid connection errors:
#      ```python
#      os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
#      os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"
#      ```

# %%
import glob
import itertools
import json
import logging
import os
import shutil
import warnings
from datetime import datetime

import cartopy.crs as ccrs
import earthpy as et
import earthpy.plot as ep
import earthaccess
import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot
import hvplot.pandas
import hvplot.xarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac_client
import rasterio as rio
import requests
import rioxarray as rxr
import rioxarray.merge as rxrmerge
import xarray as xr
from bokeh.util import logconfig
from dotenv import load_dotenv
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Supress warnings
logconfig.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

"""
Loading these environment variables:
    EARTHDATA_USERNAME
    EARTHDATA_PASSWORD
    GDAL_HTTP_MAX_RETRY=5
    GDAL_HTTP_RETRY_DELAY=1
    GDAL_HTTP_COOKIEFILE='~/cookies.txt'
    GDAL_HTTP_COOKIEJAR='~/cookies.txt'
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS='TIF'
    GDAL_HTTP_UNSAFESSL='YES'
"""
load_dotenv() # Load .env file

data_dir = os.path.join(et.io.HOME, et.io.DATA_NAME)
cache_dir = os.path.join(data_dir, 'clustering-kmeans')

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# %%
username = os.getenv('EARTHDATA_USERNAME')
password = os.getenv('EARTHDATA_PASSWORD')

# earthaccess.login doesn't seem to be creating this file
netrc_fpath = os.path.expanduser("~/.netrc")
if not netrc_fpath:
    with open(netrc_fpath, 'w') as file:
        file.write(f"machine urs.earthdata.nasa.gov\nlogin {username}\npassword {password}\n")

# %% [markdown]
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

# %%
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

# %% [markdown]
# ## STEP 2: STUDY SITE

# %% [markdown]
# For this analysis, you will use a watershed from the [Water Boundary Dataset](https://www.usgs.gov/national-hydrography/access-national-hydrography-products), HU12 watersheds (WBDHU12.shp).
# 
# YOUR TASK:
#   1. Download the Water Boundary Dataset for region 8 (Mississippi)
#   2. Select watershed 080902030506
#   3. Generate a site map of the watershed

# %%
def find_file(pattern, path):
    return glob.glob(f'{path}/**/{pattern}', recursive=True)
def copy_file(src, dst):
    shutil.copy(src, dst)


wbdhu12_shp = os.path.join(cache_dir, 'WBDHU12.shp')

if not os.path.exists(wbdhu12_shp):
    wbd_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Hydrography/WBD/HU2/Shape"
        "/WBD_08_HU2_Shape.zip"
    )
    wbd_path = et.data.get_data(url=wbd_url, replace=False)
    wbdhu12_files = find_file('WBDHU12*', wbd_path)
    for file in wbdhu12_files:
        copy_file(file, cache_dir)
    print(f'Copied files to {cache_dir}')
else:
    print(f'Using cached file: {wbdhu12_shp}')

# %%
gdf = gpd.read_file(wbdhu12_shp)
study_site = gdf[gdf.huc12 == '080902030506']
title = (
    f"{study_site['name'].values[0]}"
    " ("
    f"{study_site['huc12'].values[0]}"
    ")"
)
wdb_osm = study_site.hvplot(
    geo=True, 
    tiles='OSM',
    title=title,
    alpha=0.03,
    color='red'
)
wdb_osm

# %%
wdb_esri = study_site.hvplot(
    geo=True, 
    tiles='ESRI',
    title=title,
    alpha=0.03,
    color='red'
)
wdb_esri

# %% [markdown]
# I chose this watershed because it covers parts of New Orleans an is near the Mississippi Delta. Deltas are boundary areas between the land and the ocean, and tend to contain a rich variety of different land cover and land use types.
# 
# In the cell below, write a 2-3 sentence **site description** of this area that helps to put your analysis in context.

# %% [markdown]
# **SITE DESCRIPTION**
# 
# __Manuel Canal-Spanish Lake, located in Louisiana (LA), covers an extensive area of 37,355.86 acres, equivalent to 151.17 square kilometers. It's near New Orleans. Its variety of land types including dry soil, water, and vegetation makes it an ideal setting for this k-means clustering exercise.__
# 
# ![](https://media.baamboozle.com/uploads/images/510741/1651543763_75056_gif-url.gif)

# %% [markdown]
# ## STEP 3: MULTISPECTRAL DATA

# %% [markdown]
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

# %%
auth = earthaccess.login(strategy='netrc')

if not auth.authenticated:
    raise ValueError('Authentication failed')

# %%
granules = earthaccess.search_data(
    short_name='HLSL30',
    cloud_hosted=True,
    bounding_box=tuple(study_site.total_bounds),
    temporal=('May 2023', 'October 2023')
) # DataGranule - Dictionary-like object

# %%
granules_df = pd.json_normalize(granules)
granules_df.to_csv(os.path.join(cache_dir, 'HLSL30_search_results.csv'))

# %% [markdown]
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

# %%
def create_polygon(granule_id, data):
    """
    Create a shapely Polygon a list of points embedded in a dictionary.
    """
    if len(data) > 1:
        raise ValueError(f'More than one data entry in the granule {granule_id}')
    try:
        points = data[0]['Boundary']['Points']
        coordinates = [(point['Longitude'], point['Latitude']) for point in points]
        return Polygon(coordinates)
    except:
        print(f'Error creating polygon for granule {granule_id}')
        print(json.dumps(data))
        raise

# %%
granules_simplified = []
for granule in granules:
    granule_id = granule['umm']['GranuleUR']
    datetime = granule['umm']['DataGranule']['ProductionDateTime']
    geometry = create_polygon(
        granule_id,
        granule['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons']
    )
    file_handler = earthaccess.open([granule]) # EarthAccessFile
    for item in file_handler:
        url = item.info()['name']
        filename = url.split('/')[-1]
        tile_id = filename.split('.')[2]
        band = filename.split('.')[-2]
        granules_simplified.append({
            'granule_id': granule_id,
            'datetime': datetime,
            'date': datetime.split('T')[0],
            'geometry': geometry,
            'file_handler': file_handler,
            'url': url,
            'filename': filename,
            'tile_id': tile_id,
            'band': band
        })

# %%
granules_gdf = gpd.GeoDataFrame(granules_simplified, geometry='geometry')
granules_gdf.to_csv(os.path.join(cache_dir, 'HLSL30_simplified.csv'), index=False)

# %%
# B07: SWIR 2 2.11 – 2.29 μm
# B06: SWIR 1 1.57 – 1.65 μm
# B05: NIR 0.85 – 0.88 μm
# B04: Red 0.64 – 0.67 μm
# B03: Green 0.53 – 0.59 μm
# B02: Blue 0.45 – 0.51 μm
# Fmask: QA Band
filtered_gdf = granules_gdf[granules_gdf['band'].isin(['B07', 'B06', 'B05', 'B04', 'B03', 'B02', 'Fmask'])]
filtered_gdf.to_csv(os.path.join(cache_dir, 'HLSL30_filtered.csv'), index=False)

# %%
for granule in granules:
    fpath = os.path.join(cache_dir, granule['umm']['GranuleUR'])
    if not os.path.exists(fpath):
        earthaccess.download([granule], fpath)

# %% [markdown]
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

# %%
def decode_qa(bit_string):
    """
    Decode an 8-bit string into its quality assessment components.

    Bits are listed from the MSB (bit 7) to the LSB (bit 0): 
        7-6    aerosol:
            00 - climatology
            01 - low
            10 - average
            11 - high
        5      water
        4      snow/ice
        3      cloud shadow
        2      adjacent to cloud
        1      cloud
        0      cirrus cloud
    
    Parameters:
    bit_string (str): An 8-bit string representing quality assessment flags.
    
    Returns:
    dict: A dictionary with human-readable descriptions of the QA flags.
    """
    # Define the bit positions based on the provided table
    aerosol_level_bits = bit_string[:2]
    water_bit = bit_string[2]
    snow_ice_bit = bit_string[3]
    cloud_shadow_bit = bit_string[4]
    adjacent_cloud_shadow_bit = bit_string[5]
    cloud_bit = bit_string[6]
    
    # Map the bit values to their meanings
    aerosol_levels = {
        '00': 'Climatology aerosol',
        '01': 'Low aerosol',
        '10': 'Moderate aerosol',
        '11': 'High aerosol'
    }
    
    # Decode the bits using the mapping
    aerosol_level = aerosol_levels[aerosol_level_bits]
    water = 'Yes' if water_bit == '1' else 'No'
    snow_ice = 'Yes' if snow_ice_bit == '1' else 'No'
    cloud_shadow = 'Yes' if cloud_shadow_bit == '1' else 'No'
    adjacent_cloud_shadow = 'Yes' if adjacent_cloud_shadow_bit == '1' else 'No'
    cloud = 'Yes' if cloud_bit == '1' else 'No'
    
    # Create a dictionary of the decoded values
    decoded_values = {
        'Aerosol Level': aerosol_level,
        'Water': water,
        'Snow/Ice': snow_ice,
        'Cloud Shadow': cloud_shadow,
        'Adjacent to Cloud/Shadow': adjacent_cloud_shadow,
        'Cloud': cloud
    }
    
    return decoded_values

# %%
water_bit = 2
snow_ice_bit = 3
cloud_shadow_bit = 4
adjacent_cloud_shadow_bit = 5
cloud_bit = 6
mask_bits = [cloud_bit]
temp_gdf = gpd.GeoDataFrame()
processed_granules = []

for date, gdf_by_date in filtered_gdf.groupby('date'):
    print(f'Processing date: {date}')
    fmask_das = []
    red_das = []
    green_das = []
    blue_das = []
    nir_das = []
    swir1_das = []
    swir2_das = []
    merged_fmask = []
    merged_red = []
    merged_green = []
    merged_blue = []
    merged_nir = []
    merged_swir1 = []
    merged_swir2 = []
    for band, gdf_by_band in gdf_by_date.groupby('band'):
        # print(f'Processing band: {band}')
        for index, row in gdf_by_band.iterrows():
            filename = row.filename
            folder = row.granule_id
            filepath = os.path.join(cache_dir, folder, filename)
            # print(filepath, row.band)
            # Set masked=True to convert -9999 to NaN
            if row.band == 'B02':
                blue_da = rxr.open_rasterio(filepath, masked=True)
                blue_da.name = 'B02_Blue'
                blue_das.append(blue_da)
            elif row.band == 'B03':
                green_da = rxr.open_rasterio(filepath, masked=True)
                green_da.name = 'B03_Green'
                green_das.append(green_da)
            elif row.band == 'B04':
                red_da = rxr.open_rasterio(filepath, masked=True)
                red_da.name = 'B04_Red'
                red_das.append(red_da)
            elif row.band == 'B05':
                nir_da = rxr.open_rasterio(filepath, masked=True)
                nir_da.name = 'B05_NIR'
                nir_das.append(nir_da)
            elif row.band == 'B06':
                swir1_da = rxr.open_rasterio(filepath, masked=True)
                swir1_da.name = 'B06_SWIR1'
                swir1_das.append(swir1_da)
            elif row.band == 'B07':
                swir2_da = rxr.open_rasterio(filepath, masked=True)
                swir2_da.name = 'B07_SWIR2'
                swir2_das.append(swir2_da)
            elif row.band == 'Fmask':
                fmask_da = rxr.open_rasterio(filepath, masked=True)
                fmask_das.append(fmask_da)
    merged_blue = rxrmerge.merge_arrays(blue_das)
    merged_green = rxrmerge.merge_arrays(green_das)
    merged_red = rxrmerge.merge_arrays(red_das)
    merged_nir = rxrmerge.merge_arrays(nir_das)
    merged_swir1 = rxrmerge.merge_arrays(swir1_das)
    merged_swir2 = rxrmerge.merge_arrays(swir2_das)
    merged_fmask = rxrmerge.merge_arrays(fmask_das)

    aoi = study_site.to_crs(merged_red.rio.crs)

    clipped_fmask = merged_fmask.rio.clip_box(*aoi.total_bounds)
    clipped_red = merged_red.rio.clip_box(*aoi.total_bounds)
    clipped_green = merged_green.rio.clip_box(*aoi.total_bounds)
    clipped_blue = merged_blue.rio.clip_box(*aoi.total_bounds)
    clipped_nir = merged_nir.rio.clip_box(*aoi.total_bounds)
    clipped_swir1 = merged_swir1.rio.clip_box(*aoi.total_bounds)
    clipped_swir2 = merged_swir2.rio.clip_box(*aoi.total_bounds)

    bits = np.unpackbits(clipped_fmask.values.astype(np.uint8)).reshape(clipped_fmask.shape + (-1,))
    cloud_mask = np.prod(bits[..., mask_bits] == 0, axis=-1)

    masked_red = clipped_red.where(cloud_mask)
    masked_green = clipped_green.where(cloud_mask)
    masked_blue = clipped_blue.where(cloud_mask)
    masked_nir = clipped_nir.where(cloud_mask)
    masked_swir1 = clipped_swir1.where(cloud_mask)
    masked_swir2 = clipped_swir2.where(cloud_mask)
    processed_granules.append({
        'date': date,
        'red': masked_red,
        'green': masked_green,
        'blue': masked_blue,
        'nir': masked_nir,
        'swir1': masked_swir1,
        'swir2': masked_swir2
    })

# %%
output_folder = 'matplotlib'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for processed_granule in processed_granules:
    red_array = np.squeeze(processed_granule['red'].values)
    green_array = np.squeeze(processed_granule['green'].values)
    blue_array = np.squeeze(processed_granule['blue'].values)

    red_array = np.nan_to_num(red_array, nan=0.0)
    green_array = np.nan_to_num(green_array, nan=0.0)
    blue_array = np.nan_to_num(blue_array, nan=0.0)

    rgb_image = np.stack([red_array, green_array, blue_array], axis=-1)

    if np.issubdtype(rgb_image.dtype, np.floating):
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    else:
        raise

    plt.imsave(f'{output_folder}/{processed_granule["date"]}.png', rgb_image)

# %%
output_folder = 'hvplot'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for processed_granule in processed_granules:
    # Stack the RGB bands to create a multi-dimensional xarray DataArray
    rgb_values = ['B04_Red', 'B03_Green', 'B02_Blue']
    rgb_channel = xr.DataArray(rgb_values, dims='rgb_channel', name='rgb_channel')
    rgb_stacked = xr.concat([
        processed_granule['red'], 
        processed_granule['green'], 
        processed_granule['blue']
    ], dim=rgb_channel)
    # Ensure that the data type is appropriate for visualization
    # Normalize if the data is in float format, or ensure it's within [0, 255] if it's integer
    if np.issubdtype(rgb_stacked.dtype, np.floating):
        rgb_stacked = (rgb_stacked - rgb_stacked.min()) / (rgb_stacked.max() - rgb_stacked.min()) * 255
    rgb_stacked = rgb_stacked.astype(np.uint8)
    title = f'Manuel Canal-Spanish Lake: {processed_granule["date"]}'
    rgb_plot = rgb_stacked.hvplot.rgb(
                        x='x', y='y',
                        xlabel='Latitude', 
                        ylabel='Longitude',
                        bands='rgb_channel', 
                        data_aspect=1,
                        title=title,
                        crs=ccrs.UTM(16))
    hvplot.save(rgb_plot, f'{output_folder}/{processed_granule["date"]}.png', fmt='png')

# %% [markdown]
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

# %%
red_das, green_das, blue_das = [], [], []
nir_das, swir1_das, swir2_das = [], [], []
ignore_dates = ["2023-05-22",
                "2023-06-01",
                "2023-06-23",
                "2023-07-17",
                "2023-08-02",
                "2023-08-18",
                "2023-08-26",
                "2023-09-03"]
for processed_granule in processed_granules:
    date = processed_granule['date']
    if date in ignore_dates:
        continue
    print(f"Processing date: {date}")
    red_da = processed_granule['red']
    green_da = processed_granule['green']
    blue_da = processed_granule['blue']
    nir_da = processed_granule['nir']
    swir1_da = processed_granule['swir1']
    swir2_da = processed_granule['swir2']

    date_dimension = xr.DataArray([date], dims='date')
    expanded_red_da = red_da.expand_dims(date=date_dimension)
    expanded_green_da = green_da.expand_dims(date=date_dimension)
    expanded_blue_da = blue_da.expand_dims(date=date_dimension)
    expanded_nir_da = nir_da.expand_dims(date=date_dimension)
    expanded_swir1_da = swir1_da.expand_dims(date=date_dimension)
    expanded_swir2_da = swir2_da.expand_dims(date=date_dimension)

    red_das.append(expanded_red_da)
    green_das.append(expanded_green_da)
    blue_das.append(expanded_blue_da)
    nir_das.append(expanded_nir_da)
    swir1_das.append(expanded_swir1_da)
    swir2_das.append(expanded_swir2_da)

    if date == "2023-09-27":
        break
combined_red_da = xr.concat(red_das, dim='date')
combined_green_da = xr.concat(green_das, dim='date')
combined_blue_da = xr.concat(blue_das, dim='date')
combined_nir_da = xr.concat(nir_das, dim='date')
combined_swir1_da = xr.concat(swir1_das, dim='date')
combined_swir2_da = xr.concat(swir2_das, dim='date')

# %%
mean_red = combined_red_da.mean(dim='date')
mean_green = combined_green_da.mean(dim='date')
mean_blue = combined_blue_da.mean(dim='date')
mean_nir = combined_nir_da.mean(dim='date')
mean_swir1 = combined_swir1_da.mean(dim='date')
mean_swir2 = combined_swir2_da.mean(dim='date')


mean_red = mean_red.squeeze('band', drop=True)
mean_green = mean_green.squeeze('band', drop=True)
mean_blue = mean_blue.squeeze('band', drop=True)
mean_nir = mean_nir.squeeze('band', drop=True)
mean_swir1 = mean_swir1.squeeze('band', drop=True)
mean_swir2 = mean_swir2.squeeze('band', drop=True)
rgb_bands = [4, 3, 2]
rgb_bands = xr.DataArray(rgb_bands, dims='band', name='band')
bands = [2,3,4,5,6,7]
bands = xr.DataArray(bands, dims='band', name='band')

# Stacking the RGB bands to create a multi-dimensional xarray DataArray
rgb_stacked = xr.concat([
        mean_red, 
        mean_green, 
        mean_blue
    ], dim=rgb_bands)
rgb_stacked.name = 'RGB'
bands_stacked = xr.concat([
        mean_blue,
        mean_green,
        mean_red,
        mean_nir,
        mean_swir1,
        mean_swir2
    ], dim=bands)


rgb_stacked = (rgb_stacked - rgb_stacked.min()) / (rgb_stacked.max() - rgb_stacked.min()) * 255
rgb_stacked = rgb_stacked.astype(np.uint8)
title = f'Manuel Canal-Spanish Lake: Composite Image'
rgb_plot = rgb_stacked.hvplot.rgb(
                    x='x', y='y',
                    xlabel='Latitude', 
                    ylabel='Longitude',
                    bands='band', 
                    data_aspect=1,
                    title=title,
                    crs=ccrs.UTM(16))
rgb_plot

# %%
fig, ax = plt.subplots()

rgb_image = np.stack([mean_red, mean_green, mean_blue], axis=-1)
rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

ax.imshow(rgb_image)
title = f'Manuel Canal-Spanish Lake: Composite Image'
ax.set_title(title)
ax.set_aspect('equal')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# %% [markdown]
# ## STEP 4: K-MEANS

# %% [markdown]
# Cluster your data by spectral signature using the k-means algorithm. 
# 
# **YOUR TASK:**
# 1. Convert your DataArray into a **tidy** DataFrame of reflectance values (hint: check out the `.unstack()` method)
# 2. Filter out all rows with no data (all 0s or any N/A values)
# 3. Fit a k-means model. You can experiment with the number of groups to find what works best.

# %%
k_means_df = (
    rgb_stacked
    .drop('spatial_ref')
    .to_dataframe(name='reflectance')
    .unstack('band')
)
k_means_df

# %%
processed_xr = rgb_stacked
# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction = KMeans(n_clusters=3).fit_predict(k_means_df.values)

# Reshaping the prediction array to match the original raster shape
prediction = prediction.reshape(processed_xr['band'==4].shape)
prediction

# %% [markdown]
# ## STEP 5: PLOT

# %% [markdown]
# **YOUR TASK:**
# Create a plot that shows the k-means clusters next to an RGB image of the area. You may need to brighted your RGB image by multiplying it by 10.

# %%
# Unsupervised classification means we don't know what the groups are
numbers = [1, 2, 3]
groups = [f"Group {number}" for number in numbers]

def plot_classified_data(processed_xr, prediction, groups):
    # Plotting the classified data next to the original data
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ep.plot_rgb(processed_xr.values,
                rgb=[0, 1, 2],
                ax=ax1)
    ax1.set_title("Composite RGB Image")

    im2 = ax2.imshow(prediction,
                     cmap="viridis",
                     interpolation='none')
    ax2.axis('off')
    ep.draw_legend(im_ax=im2,
                   titles=groups)
    ax2.set_title("K-Means Classified Image")

    plt.show()
plot_classified_data(processed_xr, prediction, groups)

# %%
spectral_df = (
    bands_stacked
    .drop('spatial_ref')
    .to_dataframe(name='reflectance')
    .unstack('band')
)
# Drop the 'reflectance' level
spectral_df.columns = spectral_df.columns.droplevel()
spectral_df = spectral_df.reset_index(drop=True)
spectral_df.columns = spectral_df.columns.astype(str)
spectral_df.to_csv(os.path.join(cache_dir, 'spectral_df.csv'))

plt.scatter(spectral_df['4'], spectral_df['5'])
plt.xlabel('B04 Red')
plt.ylabel('B05 NIR')
plt.title('B05 NIR vs B04 Red')
plt.show()

# %% [markdown]
# ![NIR vs Red](https://www.mdpi.com/sensors/sensors-21-03457/article_deploy/html/images/sensors-21-03457-g001-550.jpg)
# 
# When comparing these two images above, we can start with 3 clusters and 3 primary land types: water, vegetation, and dry soil.

# %%
groups = ["Water", "Vegetation", "Dry Soil"]
plot_classified_data(processed_xr, prediction, groups)

# %%
wdb_osm + wdb_esri

# %% [markdown]
# **Plotting B04 Red against B05 NIR, we observe that the data points form a triangle, suggesting the presence of three primary land types: water, vegetation, and dry soil. Overall, the K-Means Classified Image performs quite well, with the notable exception of Lake Lery not being identified. This omission may stem from the method used to create the composite image. When comparing the K-Means Classified Image against the Open Street Map and the ESRI Map, there is generally good alignment.**

# %% [markdown]
# **YOUR PLOT HEADLINE AND DESCRIPTION HERE**
# 
# ![](https://media.baamboozle.com/uploads/images/510741/1651543763_75056_gif-url.gif)

# %%
spectral_df

# %%
dfs = []
lookup_wavelength = {
    "2": "0.45–0.51",
    "3": "0.53–0.59",
    "4": "0.64–0.67",
    "5": "0.85–0.88",
    "6": "1.57–1.65",
    "7": "2.11–2.29"
}

for column in spectral_df.columns:
    wavelength = lookup_wavelength[f'{column}']
    df = pd.DataFrame()
    df[f'Band {column}'] = spectral_df[column]
    df['Wavelength'] = wavelength
    df = df[['Wavelength', f'Band {column}']]
    dfs.append(df)

# %%
import plotly.graph_objects as go

fig = go.Figure()

for df in dfs:
    # Using the first value in 'category' column as the name for this trace
    fig.add_trace(go.Scatter(x=df[df.columns[0]], y=df[df.columns[1]], mode='lines', name=df.columns[1]))

fig.update_layout(
    title='HLSL30: Manuel Canal-Spanish Lake',
    xaxis_title='Wavelegth (μm)',
    yaxis_title='Reflectance',
)

fig.show()

# %%
fig, ax = plt.subplots()

for df in dfs:
    ax.plot(df[df.columns[0]], df[df.columns[1]], label=df.columns[1])

ax.set_title('HLSL30: Manuel Canal-Spanish Lake')
ax.set_xlabel('Wavelegth (μm)')
ax.set_ylabel('Reflectance')
ax.legend()
plt.show()

# %% [markdown]
# ![Spectral Plot](https://www.researchgate.net/profile/Anna-Zanchetta/publication/326405082/figure/fig2/AS:648558185291777@1531639729958/Typical-spectral-signatures-of-specic-land-cover-types-in-the-VIS-and-IR-region-of-the.png)

# %%
%%capture
%%bash
jupyter nbconvert 00-kmeans.ipynb --to html


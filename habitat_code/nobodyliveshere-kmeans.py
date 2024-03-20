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
from osgeo import gdal
print(gdal.__version__)


# %%
import os

import earthaccess as eta
import earthpy as et
import geopandas as gpd
import geoviews.tile_sources as gt
import hashlib
import hvplot.pandas 
import numpy as np
import pandas as pd
import pickle
import rioxarray as rxr
import xarray as xr


from osgeo import gdal
from shapely.geometry import Polygon
from sklearn.cluster import KMeans



os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"


# %%
import sys

# Print the Python path
for path in sys.path:
    print(path)


# %%
# Required GDAL Options
gdal.SetConfigOption('GDAL_HTTP_COOKIEFILE','~/cookies.txt')
gdal.SetConfigOption('GDAL_HTTP_COOKIEJAR', '~/cookies.txt')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR')
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS','TIF')
gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'YES')

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
def cached(key_func, override=False):

    def compute_and_cache_decorator(compute_function):
        
        def compute_and_cache(*args, **kwargs):
             # Generate a unique key using the key_func
            key = key_func(*args, **kwargs)
            # Create a hash of the key to use as a filename
            hashed_key = hashlib.md5(key.encode()).hexdigest()
            filename = os.path.join(et.io.HOME, et.io.DATA_NAME, 'jars', f'{hashed_key}.pickle')
            
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


# URL for the watershed boundaries zip file
wbd8_url = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/WBD_08_HU2_Shape.zip'

# Download the zip file
watershed_dl_path = et.data.get_data(url=wbd8_url)
wbd_shp = os.path.join(watershed_dl_path, 'Shape', 'WBDHU12.shp')
# Create a GeoDataFrame from the shapefile within the downloaded zip
wbd_gdf = gpd.read_file(wbd_shp)

print(wbd_gdf.columns)




# Select the specific watershed
watershed = wbd_gdf[wbd_gdf['huc12'] == '080902030506']

# Use hvplot to create an interactive map
plot = watershed.hvplot(geo=True, tiles='ESRI', color='teal', frame_width=500, frame_height=500)

# Display the plot
plot


# %% [markdown]
# I chose this watershed because it covers parts of New Orleans an is near the Mississippi Delta. Deltas are boundary areas between the land and the ocean, and tend to contain a rich variety of different land cover and land use types.
# 
# In the cell below, write a 2-3 sentence **site description** of this area that helps to put your analysis in context.

# %% [markdown]
# **YOUR SITE DESCRIPTION HERE**
# 
# ![](https://media.baamboozle.com/uploads/images/510741/1651543763_75056_gif-url.gif)

# %% [markdown]
# The watershed near New Orleans and the Mississippi Delta, marked by its low elevation and subtropical climate, is a critical ecological area where freshwater and marine environments merge. This region, with its unique blend of wetlands and estuaries, is shaped by the interplay of riverine inputs and oceanic influences, fostering a diverse ecosystem adapted to its warm, humid conditions.

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
#          bounding_box=tuple(gruped_granule_grouped_granule_gdf.total_bounds),
#          temporal=("...", "..."),
#      )
#      ```

# %%
eta.login

results = eta.search_data(
         short_name="HLSL30",
         cloud_hosted=True,
         bounding_box=tuple(watershed.total_bounds),
         temporal=("2023-05-01", "2023-10-01"),
    )

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
import geopandas as gpd
from shapely.geometry import Polygon
import re

def compile_granule_info(search_results):
    granules_info = []
    desired_bands = ['01','02', '03', '04', '05', '06', '07', '09']  # Bands to include

    for result in search_results:
        # Extract granule-level information from the 'umm' section
        granule_id = result['umm']['GranuleUR']
        datetime = result['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        coords = result['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
        polygon = Polygon([(pt['Longitude'], pt['Latitude']) for pt in coords])

        # Extract URLs from the RelatedUrls field
        related_urls = result['umm']['RelatedUrls']
        fmask_url = None
        band_urls = []

        for url_info in related_urls:
            if url_info['Type'] == 'GET DATA':
                if 'Fmask.tif' in url_info['URL']:
                    fmask_url = url_info['URL']
                elif url_info['URL'].endswith('.tif'):
                    band_match = re.search(r'B(\d{2})\.tif', url_info['URL'])
                    if band_match and band_match.group(1) in desired_bands:
                        band_urls.append(url_info['URL'])

        if not fmask_url:
            print(f"No Fmask URL found for granule: {granule_id}")
            continue  # Skip this granule if no Fmask URL is found

        for url in band_urls:
            tile_id = re.search(r'T(\d{2}[A-Z]{3})', url).group(1) if re.search(r'T(\d{2}[A-Z]{3})', url) else None
            band_number = int(re.search(r'B(\d{2})\.tif', url).group(1)) if re.search(r'B(\d{2})\.tif', url) else None

            band_info = {
                'granule_id': granule_id,
                'datetime': datetime,
                'geometry': polygon,
                'tile_id': tile_id,
                'band_number': band_number,
                'band_url': url,
                'fmask_url': fmask_url
            }
            granules_info.append(band_info)

    return gpd.GeoDataFrame(granules_info, geometry='geometry')




# %%
print(results[0]['umm'])

# %%
granule_gdf = compile_granule_info(results)
print(granule_gdf.columns)  # Access columns of the DataFrame

# Grouping by 'granule_id'

if 'processed_data' not in granule_gdf.columns:
    granule_gdf['processed_data'] = pd.Series(dtype=object)

# Group the DataFrame by 'granule_id'
grouped_granule_gdf = granule_gdf.groupby('granule_id')


for name, group in grouped_granule_gdf:
    print(f"Columns in group {name}: {group.columns}")
    break  # Remove this break if you want to see columns for each group


# %%
def process_band_test(band_path, fmask_path, mask_bits, scale_factor, watershed):
    # Load the Fmask data
    test_fmask_da = rxr.open_rasterio(fmask_path, masked=True)

    # Ensure the CRS of the watershed matches the CRS of the raster data
    if watershed.crs != test_fmask_da.rio.crs:
        watershed = watershed.to_crs(test_fmask_da.rio.crs)

    # Clip the Fmask data to the watershed geometry
    test_fmask_clipped = test_fmask_da.rio.clip(watershed.geometry)

    # Compute the mask
    test_bits = np.unpackbits(test_fmask_clipped.astype(np.uint8)).reshape(test_fmask_clipped.shape + (-1,))
    Test_mask = np.prod(test_bits[..., mask_bits] == 0, axis=-1)

    # Load the band data and clip it to the watershed geometry
    test_band_da = rxr.open_rasterio(band_path) * scale_factor

    test_band_clipped = test_band_da.rio.clip(watershed.geometry)


    # Apply the mask
    test_masked_band_da = test_band_clipped.where(test_fmask_clipped, np.nan)

    # Check if the mask was applied correctly
    print(f"Masked band data stats - Min: {test_masked_band_da.min().item()}, Max: {test_masked_band_da.max().item()}, Mean: {test_masked_band_da.mean().item()}, NaNs: {np.isnan(test_masked_band_da).sum().item()}")

    return test_masked_band_da


# %%
# Assuming 'band_url' is the column name where the band paths are stored
mask_bits = [1, 2, 3, 6, 7]
scale_factor = 0.0001
test_band_path = granule_gdf['band_url'].iloc[0]
test_fmask_path = granule_gdf['fmask_url'].iloc[0]
print(f"Test band path: {test_band_path}")

# %%


# %%
processed_band = process_band_test(test_band_path, test_fmask_path, mask_bits, scale_factor, watershed)


# %%
original_band = rxr.open_rasterio(test_band_path)

print("Original data:")
# Using .values to print the underlying NumPy array directly
print(original_band.shape)
print(original_band.values)


print("\nProcessed data:")
# Assuming processed_band is also an xarray.DataArray, printing its values
print(processed_band.shape)
print(processed_band.values)


# %%
import hvplot.xarray  # Import hvplot's xarray interface

# Ensure that the data is in a format that can be plotted (e.g., 2D DataArray)
original_band_2d = original_band.squeeze()  # Remove any singleton dimensions
processed_band_2d = processed_band.squeeze()

# Define a function to normalize the data for better visualization
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

# Normalize the data
original_band_normalized = normalize_data(original_band_2d)
processed_band_normalized = normalize_data(processed_band_2d)

# Use hvplot to create interactive plots
# Adjust the colormap to 'viridis' which is good for displaying continuous data
original_plot = original_band_normalized.hvplot.image(rasterize=True, title="Original Band", cmap='viridis', width=400, height=400, clim=(0, 1))
processed_plot = processed_band_normalized.hvplot.image(rasterize=True, title="Processed Band", cmap='viridis', width=400, height=400, clim=(0, 1))

# Combine the plots horizontally
combined_plot = original_plot + processed_plot

# Display the plot
combined_plot


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
import pandas as pd
import numpy as np
import rioxarray as rxr

# Define mask_bits and scale factor
mask_bits = [1, 2, 3, 6, 7]
scale_factor = 0.0001

# Assuming granule_gdf is your GeoDataFrame with granule information
# Add a new column for processed data if it doesn't exist
if 'processed_data' not in granule_gdf.columns:
    granule_gdf['processed_data'] = pd.Series(dtype=object)

# Function to process a single band
@cached(lambda granule_id, band_key, band_url, mask_bits, scale_factor, mask, watershed: f"{granule_id}-{band_key}")
def process_band(granule_iid, band_key, band_path, mask_bits, scale_factor, mask,  watershed):
    # Load the band data
    band_da = rxr.open_rasterio(band_path) * scale_factor

    # Ensure the watershed is in the same CRS as the band data
    watershed_reprojected = watershed.to_crs(band_da.rio.crs)

    # Clip and reproject the band data to the watershed
    band_clipped = band_da.rio.clip(watershed_reprojected.geometry, watershed_reprojected.crs)

    # Load, clip, and reproject the Fmask data to the watershed
    fmask_da = rxr.open_rasterio(fmask_path)
    fmask_clipped = fmask_da.rio.clip(watershed_reprojected.geometry, watershed_reprojected.crs)
    print(fmask_clipped.shape)

    # Ensure the mask is aligned with the band data
    mask = np.unpackbits(fmask_clipped.astype(np.uint8)).reshape(fmask_clipped.shape + (-1,))
    mask_aligned = np.prod(mask[..., mask_bits] == 0, axis=-1)
    if mask_aligned.shape != band_clipped.shape:
        raise ValueError("The mask and band data are not aligned. Check the spatial extents and resolutions.")

    # Apply the mask to the clipped and aligned band data
    masked_band_da = band_clipped.where(mask_aligned, other=np.nan)

    return masked_band_da




# Initialize a counter
counter = 0
# Set the number of rows to process (change this number as needed)
num_rows_to_process = len(granule_gdf)

# Iterate over each group and process the bands
for granule_id, group in granule_gdf.groupby('granule_id'):
    print(f"Processing granule: {granule_id}")
    fmask_path = group.iloc[0]['fmask_url']
    fmask_da = rxr.open_rasterio(fmask_path)
    bits = np.unpackbits(fmask_da.astype(np.uint8)).reshape(fmask_da.shape + (-1,))
    mask = np.prod(bits[..., mask_bits] == 0, axis=-1)

    for _, row in group.iterrows():
        if counter >= num_rows_to_process:
            break  # Stop processing if the counter reaches the limit

        band_number = row['band_number']
        band_url = row['band_url']
        if isinstance(band_number, float) and not np.isnan(band_number):
            band_number = int(band_number)
        if isinstance(band_number, int) and band_url.endswith('.tif') and 'B' in band_url:
            band_key = 'B' + str(band_number)
            processed_data = process_band(granule_id, band_key, band_url, mask_bits, scale_factor, mask, watershed)
            granule_gdf.at[_, 'processed_data'] = processed_data
            print(f"Processed data for {granule_id} - {band_key}: {granule_gdf.at[_, 'processed_data']}")

        counter += 1  # Increment the counter after processing each row

        if counter >= num_rows_to_process:
            break  # Break out of the inner loop

    if counter >= num_rows_to_process:
        break  # Break out of the outer loop


# %%
print("Processed data for the first two rows:")
for i in range(num_rows_to_process):
    print(granule_gdf.at[i, 'processed_data'])

# %%
import matplotlib.pyplot as plt
import numpy as np

# Function to plot a band
def plot_band(data, title):
    # Check data statistics
    print(f"Data stats - Min: {np.nanmin(data)}, Max: {np.nanmax(data)}, Mean: {np.nanmean(data)}")

    # Normalize the data for visualization
    data = data.squeeze()
    data_normalized = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    plt.figure(figsize=(6, 6))
    im = plt.imshow(data_normalized, cmap='gray')
    plt.title(title)
    plt.colorbar(im)
    plt.show()

# Assuming granule_gdf is your GeoDataFrame and it has been properly prepared
# Extract processed data for bands B02, B03, and B04
band_b02_data = granule_gdf.loc[granule_gdf['band_number'] == 2, 'processed_data'].iloc[0]
band_b03_data = granule_gdf.loc[granule_gdf['band_number'] == 3, 'processed_data'].iloc[0]
band_b04_data = granule_gdf.loc[granule_gdf['band_number'] == 4, 'processed_data'].iloc[0]

# Plot each band
plot_band(band_b02_data, "Band B02")
plot_band(band_b03_data, "Band B03")
plot_band(band_b04_data, "Band B04")


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
import pandas as pd
import xarray as xr

# Assuming granule_gdf is prepared and includes 'date', 'band_number', 'processed_data' columns

# Unique bands and dates
bands = granule_gdf['band_number'].unique()
dates = granule_gdf['datetime'].unique()

composite_data = []

for band in bands:
    band_data = []
    for date in dates:
        # Filter DataFrame for current band and date
        df_filtered = granule_gdf[(granule_gdf['band_number'] == band) & (granule_gdf['datetime'] == date)]
        
        # Prepare a list of DataArrays for current band and date
        data_arrays = [row['processed_data'] for _, row in df_filtered.iterrows()]
        
        # Merge and mask granules
        merged_data = xr.concat(data_arrays, dim='concat_dim').mean(dim='concat_dim')
        merged_data = merged_data.where(merged_data > 0)  # Mask negative values
        
        band_data.append(merged_data)
    
    # Concatenate along the date dimension and take the mean to create a composite
    band_composite = xr.concat(band_data, dim=pd.Index(dates, name='date')).mean(dim='date')
    
    # Check if 'band' is already a dimension and assign the band number accordingly
    if 'band' not in band_composite.dims:
        band_composite = band_composite.expand_dims({'band': [band]})
    else:
        band_composite['band'] = [band]  # Directly assign the band number
    
    composite_data.append(band_composite)

# Concatenate along the band dimension to create the final composite
composite_da = xr.concat(composite_data, dim='band')


# %%
import pandas as pd
import xarray as xr

# Assuming granule_gdf is prepared and includes 'date', 'band_number', 'processed_data' columns

# Unique bands and dates
bands = granule_gdf['band_number'].unique()
dates = granule_gdf['datetime'].unique()

composite_data = []

for band in bands:
    band_data = []
    print(f"Processing band: {band}")
    for date in dates:
        # Filter DataFrame for current band and date
        df_filtered = granule_gdf[(granule_gdf['band_number'] == band) & (granule_gdf['datetime'] == date)]
        
        # Prepare a list of DataArrays for current band and date
        data_arrays = [row['processed_data'] for _, row in df_filtered.iterrows()]
        
        # Print information about the DataArrays
        for i, da in enumerate(data_arrays):
            print(f"DataArray {i} info for band {band}, date {date}: shape={da.shape}, coords={da.coords}")
        
        # Merge and mask granules
        if data_arrays:
            merged_data = xr.concat(data_arrays, dim='concat_dim').mean(dim='concat_dim')
            print(f"Merged data shape for band {band}, date {date}: {merged_data.shape}")
            merged_data = merged_data.where(merged_data > 0)  # Mask negative values
            band_data.append(merged_data)
        else:
            print(f"No data available for band {band}, date {date}")
    
    if band_data:
        # Concatenate along the date dimension and take the mean to create a composite
        band_composite = xr.concat(band_data, dim=pd.Index(dates, name='date')).mean(dim='date')
        print(f"Band composite shape for band {band}: {band_composite.shape}")
        
        # Check if 'band' is already a dimension and assign the band number accordingly
        if 'band' not in band_composite.dims:
            band_composite = band_composite.expand_dims({'band': [band]})
        else:
            band_composite['band'] = [band]  # Directly assign the band number
        
        composite_data.append(band_composite)
    else:
        print(f"No data processed for band {band}")

# Concatenate along the band dimension to create the final composite
composite_da = xr.concat(composite_data, dim='band')
print(f"Final composite shape: {composite_da.shape}")


# %%
# print(composite_da)
print("Dimensions:", composite_da.dims)
print("Coordinates:", composite_da.coords)
print(composite_da.band)



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
kmeans_model_df = (
    composite_da
    .sel(band=[1, 2, 3, 4, 5, 6, 7, 9]) 
    .to_dataframe(name="composite_values")
    .unstack('band')
    .dropna()
)

# Assuming df is your DataFrame
kmeans_model_df = kmeans_model_df.drop('spatial_ref', axis=1, level=0)



# %%
# Using an elbow plot to find optimum number K
def optimize_k_means(df, max_k):
    means = []
    inertias = []
    
    # Iterate through number of clustes
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(df)

        means.append(k)
        inertias.append(kmeans.inertia_)

    #Create elbow plot
    fig = plt.subplots(figsize = (10,5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('# of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

# %%
optimize_k_means(kmeans_model_df, 10)

# %%
kmeans = KMeans(n_clusters = 5)
kmeans_model_df['Category'] = kmeans.fit_predict(kmeans_model_df) 

kmeans_model_df

# %% [markdown]
# ## STEP 5: PLOT

# %% [markdown]
# **YOUR TASK:**
# Create a plot that shows the k-means clusters next to an RGB image of the area. You may need to brighted your RGB image by multiplying it by 10.

# %%
import xarray as xr
import hvplot.xarray

# Assuming composite_da is an xarray DataArray with the reflectance values
rgb = composite_da.sel(band=[4, 3, 2])
rgb_uint8 = (rgb * 255).astype(np.uint8()) #  # Brighten the image and clip values to be between 0 and 1
rgb_mask = rgb_uint8.where(rgb!=np.max)
rgb_bright = rgb_mask * 10
rgb_sat = rgb_bright.where(rgb_bright <= 255, 255)

# %%
import xarray as xr
import hvplot.xarray
import numpy as np

# Assuming composite_da is an xarray DataArray with the reflectance values
rgb = composite_da.sel(band=[4, 3, 2])
rgb_uint8 = (rgb * 255).astype(np.uint8).clip(0, 255)
rgb_mask = rgb_uint8.where(rgb_uint8 != rgb_uint8.max())
rgb_bright = rgb_mask * 10  # Adjust this factor based on your brightness needs
rgb_sat = rgb_bright.where(rgb_bright <= 255, 255)

# Plotting
rgb_plot = rgb_sat.hvplot.rgb(y='y', x='x', bands='band', aspect=2)
category_plot = kmeans_model_df.Category.to_xarray().sortby(['x', 'y']).hvplot(x='x', y='y', aspect=1, cmap='colorblind')

# Combine the plots
combined_plot = rgb_plot + category_plot
combined_plot


# %% [markdown]
# Don't forget to interpret your plot!

# %% [markdown]
# **YOUR PLOT HEADLINE AND DESCRIPTION HERE**
# 
# ![](https://media.baamboozle.com/uploads/images/510741/1651543763_75056_gif-url.gif)



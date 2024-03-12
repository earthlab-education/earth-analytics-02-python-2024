# %% [markdown]
# # Changes in vegetation caused by the Mud Creek Landslide
# 
# ![Mud Creek Landslide](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/styles/list_item/public/thumbnails/image/BigSur1_May20SlideDES_0.jpg)
# 
# > Image source: [USGS](https://www.usgs.gov/special-topics/big-sur-landslides/science/mud-creek-landslide-may-20-2017)

# %%
import re

import earthaccess
import pandas as pd
import rioxarray as rxr

# %%
# Location and time information for the Mud Creek Landslide
lat = 35.76
lon = -121.30
min_lat = lat - .5
max_lat = lat + .5
min_lon = lon - .5
max_lon = lon + .5
bbox = (min_lon, min_lat, max_lon, max_lat)

date = '2017-05-20'
min_date = '2017-03'
max_date = '2017-07'
date_range = (min_date, max_date)

# %%
earthaccess.login()

# %%
results = earthaccess.search_data(
    short_name='HLSL30',
    cloud_hosted=True,
    bounding_box=bbox,
    temporal=date_range
)

# %%
pd.to_datetime([result['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'] for result in results])

# %% [markdown]
# # Overall Goal:
# Develop a time-series of CIR images showing change over time around the Mud Creek Landslide
# 
# 
# Need: NIR, Red, Green bands of Landsat-Sentinel harmonized data
# 
# Function: open and crop one band
# 
# 1. For each date:
#   1. Get the band name
#   2. Select NIR, Red, and Green bands
#   3. For each band:
#       If the band is one of NIR, Red, or Green:
#         1. For each granule:
#           1. Open and crop the dataset
#           2. Open and crop the cloud mask
#           3. Apply a cloud mask
#         2. Merge granules
#   4. Concatenate along a new band dimension
# 2. Concatenate along the date dimension
# 
# For each date:
#   Generate a CIR image
# Create an animation

# %%
file_re = re.compile(r'HLS\.L30\.(?P<match>[A-Z0-9]+)\.[T0-9]\.v2\.0\.(?P<band>[A-Za-z0-9]+)\.tif')
band_info_df = None
for result in results:
    date = pd.to_datetime(result['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']).date()
    #files = earthaccess.open([result])
    print(files[0])
    bands = [file_re.search(file.full_name).group('band') for file in files]
    df = pd.DataFrame(dict(
        date=date,
        file_handler = files,
        band = bands,
        ))
    if band_info_df is None:
        band_info_df = df
    else:
        band_info_df = pd.concat([band_info_df, df])
    break
band_info_df

# %%
# Function: open and crop one band
def open_crop_band(image_url, bbox):
    """Open and crop a single DataArray"""
    return f"Cropped inage from {image_url}"

# For each date:
for date, date_data in result_df.groupby('date'):
    print(date)
    print(date_data)
    break
#   1. Get the band name
#   2. Select NIR, Red, and Green bands
#   3. For each band:
#       If the band is one of NIR, Red, or Green:
#         1. For each granule:
#           1. Open and crop the dataset
#           2. Open and crop the cloud mask
#           3. Apply a cloud mask
#         2. Merge granules
#   4. Concatenate along a new band dimension
# 2. Concatenate along the date dimension

# For each date:
#   Generate a CIR image
# Create an animation



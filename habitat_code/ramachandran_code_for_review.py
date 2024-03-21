# %% [markdown]
# # Boulder, Colorado Urban Greenspace
# 
# Boulder, Colorado is a city in the foothills of the Front Range of the Rocky Mountains in the U.S. state of Colorado. It has a semi-arid climate with cold winters and warm, relatively wet summers. The City has been considered a leader in urban forestry being named a Tree City USA by the National Arbor Day Foundation since 1984 [1]. The city maintains a street tree inventory. In 2018, the City of Boulder released the Urban Forest Strategic Plan and in 2023, the City issued a State of the Urban Forest Report [2]. The Strategic Plan outlines a goal to maintain the city's tree canopy at 16% (a no-net-loss goal) given ongoing declines in the tree canopy due to the emerald ash borer.
# 
# In this notebook, I examined greenspace across the City of Boulder by census tract and conducted a linear regression analysis to test if median annual income correlates with the fraction of greenspace.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/7/7e/BoulderBearPeak.jpg" alt="Image of Boulder, Colorado" width="400">
# 
# > Sources:
# >
# > [1] https://bouldercolorado.gov/government/departments/forestry/about#main-content
# >
# > [2] https://storymaps.arcgis.com/stories/0cb784ee805144428f914f904a0bb367

# %% [markdown]
# ## STEP 1: Set up your analysis

# %%
import cartopy.crs as ccrs
from census import Census
import earthpy as et
import geoviews as gv
import geopandas as gpd
import getpass

import hvplot
import holoviews as hv
import hvplot.pandas
import hvplot.xarray

import io

import matplotlib.pyplot as plt
import numpy as np
import os

import pystac_client
import pandas as pd

import requests
import rioxarray as rxr
from rioxarray.merge import merge_arrays


import shapely

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import us

import xarray as xr

# %%
data_dir = os.path.join(et.io.HOME, et.io.DATA_NAME, 'boulder-greenspace')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# %%
%%bash
ls ~/earth-analytics/data/boulder-greenspace

# %%
# Download City of Boulder boundary

boundary_url = ("https://gis.bouldercolorado.gov/ags_svr1/rest/services/plan/CityLimits/MapServer/0/query?outFields=*&where=1%3D1&f=geojson")

boundary_path = os.path.join(data_dir, 'city_boundary.geojson')

if os.path.exists(boundary_path):

    boundary_gdf = gpd.read_file(boundary_path)
    print("Data is already downloaded.")

else:
    # Mimic web browser
    user_agent = (
        'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) '
        'Gecko/20100101 Firefox/81.0'
    )

    # Download GEOJSON
    r = requests.get(url=boundary_url, headers={'User-Agent': user_agent})

    # Read GeoJSON data into a GeoDataFrame
    boundary_geojson_data = r.json()

    # Cache
    boundary_gdf = (gpd.GeoDataFrame
                    .from_features(boundary_geojson_data['features'])
                    .to_file(boundary_path, driver="GeoJSON")
                    )
    
    # Read
    boundary_gdf = gpd.read_file(boundary_path)
    print("Data downloaded and loaded.")



# %%
boundary_gdf

# %%
boundary_gdf.to_crs("4326").plot()

# %%
# Download census tracts
print(us.states.CO.fips)

colorado_tracts_url = ("https://www2.census.gov/geo/tiger/TIGER2023/"
           "TRACT/tl_2023_08_tract.zip")

colorado_tracts_path = os.path.join(data_dir, 'colorado_census_tracts.geojson')

if os.path.exists(colorado_tracts_path):
    colorado_tracts = gpd.read_file(colorado_tracts_path).to_crs("4326")
    print("Data is already downloaded.")
else:
    gpd.read_file(colorado_tracts_url).to_file(colorado_tracts_path, driver="GeoJSON")
    colorado_tracts_gdf = gpd.read_file(colorado_tracts_path).to_crs("4326")

# %%
colorado_tracts.plot(edgecolor="white", linewidth=0.5)

# %%
# Select only the census tracts in Boulder
print(colorado_tracts.crs)
print(boundary_gdf.crs)

boulder_tracts_gdf = gpd.sjoin(colorado_tracts, boundary_gdf, how="inner", predicate="intersects")
boulder_tracts_gdf.plot(edgecolor="white", linewidth=0.5)

# Clip to city boundary 
boulder_city_tracts_gdf = gpd.clip(boulder_tracts_gdf, boundary_gdf)
boulder_city_tracts_gdf.plot(edgecolor="white", linewidth=0.5)

# %%
# Download census data

# B06011_001E: Estimate!!Median income in the past 12 months --!!Total:	
# Sources: https://api.census.gov/data/2019/acs/acs5/variables.html; https://pypi.org/project/census/

# Obtain Census variables from the 2019 American Community Survey at the tract level for Illinois (FIPS code: 17)
census_path = os.path.join(data_dir, 'census_data_for_colorado.csv')

if os.path.exists(census_path):
    census_df = pd.read_csv(census_path)
    print("Data is already downloaded.")

else:
    # Authenticate
    api_key = getpass.getpass('U.S. Census API Key')
    c = Census(api_key)
    c

    CO_census = c.acs5.state_county_tract(fields = ('NAME', 'B06011_001E'),
                                      state_fips = '08',
                                      county_fips = "*",
                                      tract = "*",
                                      year = 2021)
    
    tracts_census_df = (pd.DataFrame(CO_census, 
                                     columns=['NAME', 
                                              'B06011_001E', 
                                              'state', 
                                              'county', 
                                              'tract'])
                        .to_csv(census_path, index=False))
    
    census_df = pd.read_csv(census_path)


census_df = census_df.rename(columns={'B06011_001E': 'median_income'})
census_df.head()

# Merge census data with tracts

boulder_city_tracts_gdf

# %%
# Merge datasets

boulder_city_tracts_gdf = boulder_city_tracts_gdf.loc[:, ['TRACTCE', 'geometry', 'NAMELSAD']]

boulder_city_tracts_gdf['TRACTCE'] = pd.to_numeric(boulder_city_tracts_gdf['TRACTCE'], errors='coerce')

tracts_w_census_gdf = boulder_city_tracts_gdf.merge(census_df, left_on='TRACTCE', right_on='tract')

negative_or_not = (tracts_w_census_gdf['median_income'] >= 0)

tracts_w_census_gdf = tracts_w_census_gdf[negative_or_not]
tracts_w_census_gdf.head()

# %%
tracts_w_census_gdf['median_income'].plot.hist()

# %%
tracts_w_census_gdf.plot('median_income', legend=True)

# %%
# Test whether tracts are correctly georeferenced

tracts_basemap_plot = tracts_w_census_gdf.hvplot(geo=True, alpha=0.3, tiles='EsriImagery')
tracts_basemap_plot

# %%
# Download data using Microsoft Planetary Computer STAC catalog

# Access catalog
pc_catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

pc_catalog.title

# %%
# Check if any data has already been downloaded and processed

all_greenspace_stats_path = os.path.join(data_dir, 'all_greenspace_stats.csv')

if os.path.exists(all_greenspace_stats_path):
    print("All greenspace stats file exists. Checking "
          "which tracts have been downloaded and processed...")

    # Extract tracts to be calculated
    all_tracts = tracts_w_census_gdf['TRACTCE'].astype('int64')

    # Load tracts already processed
    all_greenspace_stats_df = pd.read_csv(all_greenspace_stats_path)

    # Check which values in tract_numbers are not downloaded
    missing_tracts = all_tracts[~all_tracts.isin(all_greenspace_stats_df['tract'])]

    # Create a DataFrame with the missing tracts and their geometries
    tract_geometry = tracts_w_census_gdf[['TRACTCE', 'geometry']]
    tract_geometry['TRACTCE'] = tract_geometry['TRACTCE'].astype('int64')

    missing_tracts_df = pd.DataFrame({'missing_tract': missing_tracts.astype('int64')})

    missing_tracts_merged_df = missing_tracts_df.merge(
        tract_geometry,
        left_on='missing_tract',
        right_on='TRACTCE',
        how='left'
    )

    missing_tracts_gdf = gpd.GeoDataFrame(missing_tracts_merged_df, geometry='geometry')

    missing_tracts_gdf.head()

    # Plot unprocessed tracts
    ax = missing_tracts_gdf.plot(color='purple', label='Unprocessed tracts')

    # Adding title
    plt.title('Unprocessed tracts')

    plt.show()

else:
    all_greenspace_stats_df = pd.DataFrame(columns=['tract', 'fraction_greenspace'])
    all_greenspace_stats_df.to_csv(all_greenspace_stats_path, index=False)

    print("All greenspace stats file does not exist. Created file.")
    


# %%
# Compile image URLs for all tracts that are yet to be processed

naip_image_urls_path = os.path.join(data_dir, 'naip_image_urls.csv')
year = 2021
item_url_dfs = []

# Search for image for each tract
for index, tract in missing_tracts_gdf.iterrows():
    # print(tract) 

    tract_name = tract['TRACTCE']

    print(tract_name)

    # Search catalog for image
    tract_geometry = tract['geometry']

    naip_search = pc_catalog.search(
    collections=["naip"],
    intersects=shapely.to_geojson(tract_geometry),
    datetime=f"{year}"
    )

    print(naip_search)

    try:

        for naip_item in naip_search.items():

            print(naip_item.id)
            item_url_dfs.append(
                pd.DataFrame(dict(
                    tract=[tract_name],
                    tile_id=[naip_item.id],
                    url=naip_item.assets['image'].href
                ))
            )
        
    except Exception as e:
        print(f"Error processing item: {str(e)}")
            
        continue


item_url_df = pd.concat(item_url_dfs)
item_url_df

# Save URLS
item_url_df.to_csv(naip_image_urls_path, index=False)

# %%
# Download and process data for all tracts

ndvi_threshold = 0.12

print("All greenspace stats file does not exist. Running computation.")

all_greenspace_stats = []

# Calculate greenspace fraction for each tract
for tract, tract_urls in item_url_df.groupby('tract'):

    print(tract)

    # Store all NDVI images for tract
    tract_ndvi_das = []

    try: 

        for index, image in tract_urls.iterrows():
            print("URL is:", image.url)

            # Open NAIP data array
            full_naip_vda = rxr.open_rasterio(image.url, masked=True).squeeze()

            # Get census tract boundary
            boundary_gdf = missing_tracts_gdf.to_crs(full_naip_vda.rio.crs)[missing_tracts_gdf.TRACTCE==tract]
            
            # Clip NAIP data to boundary
            crop_naip_vda = full_naip_vda.rio.clip_box(
                *boundary_gdf.total_bounds
            )

            naip_vda = crop_naip_vda.rio.clip(boundary_gdf.geometry)

            # Compute NDVI
            # Band 4: NIR, Band 1: Red
            tract_ndvi_das.append(
                (naip_vda.sel(band=4) - naip_vda.sel(band=1))
                / (naip_vda.sel(band=4) + naip_vda.sel(band=1))
            )

        # Merge rasters if there are multiple images
        if len(tract_ndvi_das)>1:
            tract_ndvi_da = merge_arrays(tract_ndvi_das)
            print("Merged images")

        else:
            print("Only one image for tract")
            tract_ndvi_da = tract_ndvi_das[0]

        # Compute fraction of greenspace (NDVI>NDVI threshold)
        fraction_greenspace_da = np.sum(tract_ndvi_da > ndvi_threshold) / tract_ndvi_da.notnull().sum()

        # Extract fraction
        if fraction_greenspace_da.size == 1:
            fraction_greenspace = fraction_greenspace_da.values.flatten()[0]
            print(fraction_greenspace)

        else:
            print("Error: The fraction greenspace array has multiple values.")

        # Add to accumulator list
        tract_stats = [tract, fraction_greenspace]
        all_greenspace_stats.append(tract_stats)
    
    except Exception as e:
            print(f"Error processing tract: {e}")
            continue  # Continue to next tract



# %%
# Join tracts with geometry

newly_processed_tract_stats = pd.DataFrame(all_greenspace_stats, columns=all_greenspace_stats_df.columns)

# Add the newly processed stats to the original df
all_greenspace_stats_new_df = pd.concat([all_greenspace_stats_df, newly_processed_tract_stats], ignore_index=True)

# Cache
all_greenspace_stats_new_df.to_csv(all_greenspace_stats_path, index=False)

# Retrieve geometries
greenspace_gdf = pd.merge(tracts_w_census_gdf[['tract', 'geometry', 'median_income', 'NAME']], all_greenspace_stats_new_df, left_on='tract', right_on='tract', how='left')
greenspace_gdf

# %%
greenspace_plot = (greenspace_gdf.hvplot(geo=True, hover_cols=['value'], cmap='viridis', c='fraction_greenspace', width=500, height=500)
                   .opts(title='Greenspace fraction by census tract in Boulder, CO')
)

median_income_plot = (tracts_w_census_gdf.hvplot(geo=True, hover_cols=['value'], cmap='viridis', c='median_income', width=500, height=500)
                   .opts(title='Median income by census tract in Boulder, CO'))

greenspace_income_plot = greenspace_plot + median_income_plot

greenspace_income_plot

# %% [markdown]
# ## Linear ordinary least-squares regression
# 

# %%
# Create df for analysis
greenspace_values = greenspace_gdf.dropna()
greenspace_values = greenspace_values[['tract', 'fraction_greenspace']]
greenspace_values['tract'] = greenspace_values['tract'].astype('int64')

income_values = tracts_w_census_gdf[['tract', 'median_income']]

analysis_df = (pd.merge(greenspace_values, income_values, on='tract', how='left')
)

analysis_df

# %%
scatter_plot = (analysis_df.hvplot.scatter(x='median_income', y='fraction_greenspace')
                       .opts(title='Median income versus greenspace fraction by census tract')
)
scatter_plot

# %%
analysis_df.hvplot.hist(y='fraction_greenspace') + analysis_df.hvplot.hist(y='median_income')

# %% [markdown]
# Fraction greenspace looks roughly normal. Median income seems skewed so we can log transform.

# %%
analysis_df['log_median_income'] = np.log(analysis_df['median_income'])

analysis_df.dropna(inplace=True)

# %%
analysis_df.hvplot.hist(y='log_median_income')

# %% [markdown]
# ## Linear regression analysis

# %%
X = analysis_df[['log_median_income']]
y = analysis_df[['fraction_greenspace']]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5, 
                                                    random_state=42)

# %%
# Fit linear regression to training data

linear_reg = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
linear_reg.coef_

y_hat = linear_reg.predict(X_test)
y_hat

test_df = y_test.copy()
test_df['y_hat'] = linear_reg.predict(X_test)
test_df['measured'] = y_test
test_df['predicted']= y_hat
y_max = float(y_test.max())

(
    test_df
    .hvplot.scatter(x='measured', y='predicted')
    .opts(aspect='equal', xlim=(0, y_max), ylim=(0, y_max), width=600, height=600)
) * hv.Slope(slope=1, y_intercept=0).opts(color='black')

# %%
# Calculate and map spatial bias in the model predictions.

analysis_df['pred_fraction_greenspace'] = linear_reg.predict(X)
analysis_df['err_fraction_greenspace'] = analysis_df['pred_fraction_greenspace'] - analysis_df['fraction_greenspace']

tract_boundaries_gdf = tracts_w_census_gdf[['tract', 'geometry']]

analysis_gdf = pd.merge(tract_boundaries_gdf, analysis_df, on='tract')
analysis_gdf

(
    analysis_gdf.hvplot(geo=True, color='err_fraction_greenspace', cmap='RdBu')
    .redim.range(err_fraction_greenspace=(-.3, .3))
    .opts(frame_width=600, aspect='equal')
)


# %% [markdown]
# The northern edge of the city seems to have a high error.



from .config import *

from . import access
import matplotlib.pyplot as plt
import math
import pandas as pd
import geopandas as gpd
import osmnx as ox

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def display_heatmap(df, title):
    plt.matshow(df)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    plt.title(title, fontsize=16)
    plt.show()

def calc_degrees_per_km(latitude):
  rad = 6371
  local_rad = rad * math.cos ( math.radians(latitude) )
  local_degrees_per_hkm = 360/(2*math.pi*local_rad)
  degrees_per_vkm = 360/(2*math.pi*rad)
  return (degrees_per_vkm, local_degrees_per_hkm)

def gen_gdf(latitude, longitude, conn):
  (degrees_per_vkm, local_degrees_per_hkm) = calc_degrees_per_km(latitude)
  rows = access.sql_select(conn, f'SELECT pp.price, pp.date_of_transfer, po.postcode, po.latitude, po.longitude, pp.property_type, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street FROM pp_data AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode WHERE Latitude >= {latitude - (degrees_per_vkm * 1)} AND Latitude <= {latitude + (degrees_per_vkm * 1)} AND Longitude >= {longitude - (local_degrees_per_hkm * 1)} AND Longitude <= {longitude + (local_degrees_per_hkm * 1)} AND date_of_transfer >= "2020-01-01" AND NOT (pp.property_type = \'F\')')
  # print(rows)
  df = pd.DataFrame(rows, columns=["Price", "Date of Transfer", "Postcode", "Latitude", "Longitude", "Type", "House Number", "Secondary Addressable Object Name", "Street"])
  gdf = gpd.GeoDataFrame(
      df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
  )
  return gdf

def get_buildings(latitude, longitude):
  (degrees_per_vkm, local_degrees_per_hkm) = calc_degrees_per_km(latitude)
  north = latitude + (degrees_per_vkm * 1)
  south = latitude - (degrees_per_vkm * 1)
  east = longitude + (local_degrees_per_hkm * 1)
  west = longitude - (local_degrees_per_hkm * 1)
  buildings = ox.features_from_bbox(north = north, south = south, west= west, east = east, tags={'building': True})
  return buildings

def display_price_graphs(gdf, place_name):
  _, ax = plt.subplots()
  ax.set_title(f'Prices in {place_name}')
  price = gdf["Price"].apply(lambda x : math.log(x))
  x = gdf.centroid.x
  y = gdf.centroid.y
  ax.scatter(x, y, c=price)
  plt.show()
  small_df = pd.DataFrame.from_dict({"price" : price, "Longitude" : x, "Latitude" : y})
  display_heatmap(small_df.corr(), f'Correlation Matrix for Latitude and Longitude against prices in {place_name}')

def do_analysis(latitude, longitude, place_name, conn):
  pp_gdf = gen_gdf(latitude, longitude, conn)
  osm_gdf=get_buildings(latitude, longitude)
  osm_gdf['Street'] = osm_gdf['addr:street'].apply(lambda x : str.upper(str(x)))
  join_df = pd.merge(pp_gdf, osm_gdf, left_on=['House Number', "Street"], right_on=['addr:housenumber', 'Street'], how='inner')
  join_gdf = join_df.set_geometry("geometry_y")
  display_price_graphs(join_gdf, place_name)
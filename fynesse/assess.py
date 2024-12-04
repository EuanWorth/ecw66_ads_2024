from .config import *

from . import access
import matplotlib.pyplot as plt
import math
import pandas as pd
import geopandas as gpd
import osmnx as ox
import statsmodels.api as sm
import numpy as np
import scipy.stats as sps


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

def display_heatmap(df, title, ax=None, use_rows=False):
    if ax == None:
      plt.matshow(df)
      if use_rows:
         plt.yticks(np.arange(len(df.index)), list(df.index), fontsize=14)
      else:
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
      plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
      plt.title(title, fontsize=16)
      plt.colorbar()
      plt.show()
    else:
      im = ax.imshow(df)
      ax.set_xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
      if use_rows:
         ax.set_yticks(np.arange(len(df.index)), list(df.index), fontsize=14)
      else:
        ax.set_yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
      ax.set_title(title, fontsize=16)
      plt.colorbar(im, ax=ax)
      



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
  pp_gdf = access.gen_gdf(latitude, longitude, conn)
  osm_gdf= access.get_buildings(latitude, longitude)
  osm_gdf['Street'] = osm_gdf['addr:street'].apply(lambda x : str.upper(str(x)))
  join_df = pd.merge(pp_gdf, osm_gdf, left_on=['House Number', "Street"], right_on=['addr:housenumber', 'Street'], how='inner')
  join_gdf = join_df.set_geometry("geometry_y")
  display_price_graphs(join_gdf, place_name)

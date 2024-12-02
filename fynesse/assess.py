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
      plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
      plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
      plt.title(title, fontsize=16)
      plt.colorbar()
      plt.show()
    else:
      im = ax.imshow(df)
      ax.set_xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
      if use_rows:
         ax.set_yticks(range(df.index).shape[1], df.index, fontsize=14)
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

def cross_validate(design_matrix, response_vector, k, n = 10, regularised = False, alpha = None, L1_wt = 1):
  test_scores = []
  if alpha == None:
    n, p = design_matrix.shape
    alpha = 1.1 * np.sqrt(n) * sps.norm.ppf(1 - 0.05 / (2 * p))
  for _ in range(n):
    permutation = np.random.permutation(len(design_matrix))
    design_matrix_folds = [design_matrix[permutation][i::k] for i in range(k)]
    response_vector_folds = [response_vector[permutation][i::k] for i in range(k)]
    for i in range(k):
      train_design_matrix = np.concatenate(design_matrix_folds[:i] + design_matrix_folds[i+1:])
      train_response_vector = np.concatenate(response_vector_folds[:i] + response_vector_folds[i+1:])
      test_design_matrix = design_matrix_folds[i]
      test_response_vector = response_vector_folds[i]
      model = sm.OLS(train_response_vector, train_design_matrix*1)
      fit = 0
      if regularised:
        fit = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
      else:
        fit = model.fit()
      predictions = fit.predict(train_design_matrix)
      rsum = np.sum((predictions - train_response_vector)**2)
      tss = np.sum((train_response_vector - np.mean(train_response_vector))**2)
      rsquared = 1 - rsum/tss
      test_scores.append(rsquared)
  return sum(test_scores)/len(test_scores)

import csv
import pymysql
import requests
import matplotlib.pyplot as plt
import math
import pandas as pd
import geopandas as gpd
import osmnx as ox
import zipfile
import io
import os
import yaml
from ipywidgets import interact_manual, Text, Password

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with 
outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond 
the legal side also think about the ethical issues around this data. """


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


def hello_world():
    print("Hello from the data science library!")


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to + 1)):
        print(f"Downloading data for year: {year}")
        for part in range(1, 3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to + 1)):
        print(f"Downloading data for year: {year}")
        for part in range(1, 3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def housing_upload_join_data(conn, year):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(
        f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print('Storing data for year: ' + str(year))
    cur.execute(
        f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    print('Data stored for year: ' + str(year))

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def sql_select(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    return cur.fetchall()

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    poi_types = ["amenity", "historic", "leisure", "shop", "tourism", "religion", "memorial"]
    pois = ox.features_from_point((latitude, longitude), tags, dist=distance_km*1000)
    poi_counts = {}

    for tag in poi_types:
      if tag in pois.columns:
        poi_counts[tag] = pois[tag].notnull().sum()
      else:
        poi_counts[tag] = 0

    return poi_counts

def calc_degrees_per_km(latitude):
  rad = 6371
  local_rad = rad * math.cos ( math.radians(latitude) )
  local_degrees_per_hkm = 360/(2*math.pi*local_rad)
  degrees_per_vkm = 360/(2*math.pi*rad)
  return (degrees_per_vkm, local_degrees_per_hkm)

def gen_gdf(latitude, longitude, conn):
  (degrees_per_vkm, local_degrees_per_hkm) = calc_degrees_per_km(latitude)
  rows = sql_select(conn, f'SELECT pp.price, pp.date_of_transfer, po.postcode, po.latitude, po.longitude, pp.property_type, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street FROM pp_data AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode WHERE Latitude >= {latitude - (degrees_per_vkm * 1)} AND Latitude <= {latitude + (degrees_per_vkm * 1)} AND Longitude >= {longitude - (local_degrees_per_hkm * 1)} AND Longitude <= {longitude + (local_degrees_per_hkm * 1)} AND date_of_transfer >= "2020-01-01" AND NOT (pp.property_type = \'F\')')
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

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")

def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

def store_credentials():
    @interact_manual(username=Text(description="Username:"),
                password=Password(description="Password:"),
                url=Text(description="URL:"),
                port=Text(description="Port:"))
    def write_credentials(username, password, url, port):
        with open("credentials.yaml", "w") as file:
            credentials_dict = {'username': username,
                            'password': password,
                            'url': url,
                            'port': port}
            yaml.dump(credentials_dict, file)

def load_credentials():
    with open("credentials.yaml") as file:
        return yaml.safe_load(file)
    

def count_pois(poi_df, poi_types = ["amenity", "historic", "leisure", "shop", "tourism", "religion", "memorial"]):
    poi_counts = {}
    for tag in poi_types:
      if tag in poi_df.columns:
        poi_counts[tag] = poi_df[tag].notnull().sum()
      else:
        poi_counts[tag] = 0
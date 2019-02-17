from yelpapi import YelpAPI
import pandas as pd
import argparse
from pprint import pprint
import math
import json

my_api = '98yxvQMvwhlOM-OM6AoxB7jqgc8VVI2vCOeZ2EHhGOtKor_uXmSpSd600-lnLMIfVEIFyuBXvjetd9QgKoqCZ1j_9bq4Rf0ylJvZ3vPXydS7gOU-d-UvIiglXAQRXHYx'

yelp_api = YelpAPI(my_api)

MIN_LAT  = 53.39341
MAX_LAT  = 53.73975
MIN_LONG = 9.730138
MAX_LONG = 10.33404
LIMIT    = 50

def get_restaurants(latitude, longitude, radius):
  offset = 0
  print('get_restaurants( lat = ' + str(latitude) + ', long = ' + str(longitude) + ', radius = ' + str(radius) + ')')
  response = yelp_api.search_query(term='restaurants', latitude=latitude, longitude=longitude, limit=50, offset=0, radius = radius)
  number_of_restaurants = response['total']
  restaurants_list = []
  while number_of_restaurants > 0:
    response_len = len(response['businesses'])
    restaurants_list.extend(response['businesses'])
    number_of_restaurants = number_of_restaurants - response_len
    offset = offset + response_len
    print('get_restaurants( lat = ' + str(latitude) + ', long = ' + str(longitude) + ', offset = ' + str(offset) + ', radius = ' + str(radius) + ')')
    if (offset + 50) < 1000:
        response = yelp_api.search_query(term='restaurants', latitude=latitude, longitude=longitude, limit=50, offset=offset, radius = radius)
    else:
        return restaurants_list
  return restaurants_list



#restaurants = get_restaurants(MIN_LAT, MIN_LONG)

import os
#os.chdir('C:\Users\vince_000\Documents\Geotesting\Test_Files\Hamburg')
import csv

filename = '0.csv'
keys = restaurants[0].keys()

with open(filename, 'w') as f:
  w = csv.DictWriter(f, keys)
  w.writeheader()
  w.writerows(restaurants)

import json
filename = '0.json'
with open(filename, 'w') as f:
  json.dump(restaurants, f)

with open(filename, 'r') as f:
  jdata = json.load(f)

import pyproj
import numpy as np

def transform_coord(lat_in, long_in): #output to yelp
  inProj = pyproj.Proj(init='epsg:25832')
  outProj = pyproj.Proj(init='epsg:4326')
  lat_out,long_out = pyproj.transform(inProj,outProj,lat_in,long_in)
  return lat_out, long_out

lat_in, long_in = 5916565, 548145
lat_out, long_out = transform_coord(lat_in, long_in)

MAX_LAT_25832 = 588050
MAX_LONG_25832 = 5955199

MIN_LAT_25832 = 548145
MIN_LONG_25832 = 5916563

TILE_SIZE = 2000

latitude, longitude = MIN_LAT_25832, MIN_LONG_25832
latitudes = []
longitudes = []
etrs_latitudes = []
etrs_longitudes = []

all_restaurants = []


while (latitude < (MAX_LAT_25832 - (TILE_SIZE/2))):
  longitude = MIN_LONG_25832
  while (longitude < (MAX_LONG_25832 - (TILE_SIZE/2))):
    etrs_latitudes.append(latitude)
    etrs_longitudes.append(longitude)
    wgs_lat, wgs_long = transform_coord(latitude, longitude)
    longitudes.append(wgs_lat)
    latitudes.append(wgs_long)
    longitude = longitude + TILE_SIZE
    
  latitude = latitude + TILE_SIZE

import pandas as pd
df_points = pd.DataFrame({'lat': latitudes, 'long': longitudes})
df_etrs_points = pd.DataFrame({'lat': etrs_latitudes, 'long': etrs_longitudes})
print(df_points)

radius = int(math.sqrt(math.pow(TILE_SIZE,2)*2))

for i in range(len(df_points)):
    lat = df_points.iloc[i,0]
    long = df_points.iloc[i,1]
    tile_restaurants = get_restaurants(lat, long, radius)
    response = json.dumps(tile_restaurants)
    filename = str(int(lat*1000)) + '_' + str(int(long*1000)) + '.json'
    f = open(filename,"w")
    f.write(response)
    f.close()







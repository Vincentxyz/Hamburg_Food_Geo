import pandas as pd
import math
import numpy as np
import pyproj
from yelpapi import YelpAPI
import json

df_big_tiles = pd.read_csv('big_tiles.csv')

my_api = '98yxvQMvwhlOM-OM6AoxB7jqgc8VVI2vCOeZ2EHhGOtKor_uXmSpSd600-lnLMIfVEIFyuBXvjetd9QgKoqCZ1j_9bq4Rf0ylJvZ3vPXydS7gOU-d-UvIiglXAQRXHYx'

yelp_api = YelpAPI(my_api)

def transform_coord(lat_in, long_in): #output to yelp
  inProj = pyproj.Proj(init='epsg:25832')
  outProj = pyproj.Proj(init='epsg:4326')
  lat_out,long_out = pyproj.transform(inProj,outProj,lat_in,long_in)
  return lat_out, long_out

def get_number_of_restaurants(latitude, longitude, radius):
  offset = 0
  print('get_restaurants( lat = ' + str(latitude) + ', long = ' + str(longitude) + ', radius = ' + str(radius) + ')')
  response = yelp_api.search_query(term='restaurants', latitude=latitude, longitude=longitude, limit=50, offset=0, radius = radius)
  number_of_restaurants = response['total']
  return number_of_restaurants
  #return response
  
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


def save_response(latitude, longitude, radius):
    tile_restaurants = get_restaurants(latitude, longitude, radius)
    response = json.dumps(tile_restaurants)
    filename = str(int(latitude*1000)) + '_' + str(int(longitude*1000)) + '.json'
    f = open(filename,"w")
    f.write(response)
    f.close()


TILE_SIZE = 2000

radius = int(math.sqrt(math.pow(TILE_SIZE,2)*2)+1)
st_radius = int((radius / 4)+1)

list_number_restaurants = []

for i in range(len(df_big_tiles.index)):

    #ST = Subtile
    
    ST_1_epsg_25832_lat = df_big_tiles.iloc[i,3] - (TILE_SIZE/4)
    ST_1_epsg_25832_long = df_big_tiles.iloc[i,4] + (TILE_SIZE/4)
    ST_2_epsg_25832_lat = df_big_tiles.iloc[i,3] + (TILE_SIZE/4)
    ST_2_epsg_25832_long = df_big_tiles.iloc[i,4] + (TILE_SIZE/4)
    ST_3_epsg_25832_lat = df_big_tiles.iloc[i,3] - (TILE_SIZE/4)
    ST_3_epsg_25832_long = df_big_tiles.iloc[i,4] - (TILE_SIZE/4)
    ST_4_epsg_25832_lat = df_big_tiles.iloc[i,3] + (TILE_SIZE/4)
    ST_4_epsg_25832_long = df_big_tiles.iloc[i,4] - (TILE_SIZE/4)
    
    ST_1_wgs_long, ST_1_wgs_lat = transform_coord(ST_1_epsg_25832_lat, ST_1_epsg_25832_long)
    ST_2_wgs_long, ST_2_wgs_lat = transform_coord(ST_2_epsg_25832_lat, ST_2_epsg_25832_long)
    ST_3_wgs_long, ST_3_wgs_lat = transform_coord(ST_3_epsg_25832_lat, ST_3_epsg_25832_long)
    ST_4_wgs_long, ST_4_wgs_lat = transform_coord(ST_4_epsg_25832_lat, ST_4_epsg_25832_long)

    
    save_response(ST_1_wgs_lat, ST_1_wgs_long, st_radius)
    save_response(ST_2_wgs_lat, ST_2_wgs_long, st_radius)
    save_response(ST_3_wgs_lat, ST_3_wgs_long, st_radius)
    save_response(ST_4_wgs_lat, ST_4_wgs_long, st_radius)

#    Get number of restaurants
#    number_of_restaurants = get_number_of_restaurants(ST_1_wgs_lat, ST_1_wgs_long, st_radius)
#    list_number_restaurants.append(number_of_restaurants)
#    
#    number_of_restaurants = get_number_of_restaurants(ST_2_wgs_lat, ST_2_wgs_long, st_radius)
#    list_number_restaurants.append(number_of_restaurants)
#    
#    number_of_restaurants = get_number_of_restaurants(ST_3_wgs_lat, ST_3_wgs_long, st_radius)
#    list_number_restaurants.append(number_of_restaurants)
#    
#    number_of_restaurants = get_number_of_restaurants(ST_4_wgs_lat, ST_4_wgs_long, st_radius)
#    list_number_restaurants.append(number_of_restaurants)
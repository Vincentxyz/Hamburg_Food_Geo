#All functions work by calling yelpapi and not from the scraped restaurant database

from yelpapi import YelpAPI
import math

g_MY_API = '2Q1MBNavtdjYjhWPJgOfmyHyRIHUrn3f9zKgSuInJ0x3BB95Z9CJRR7-TS6Q0agTTOwq5YwWa6NbfVsmXyiXGHmDGd2E4L9q676rDDw3c8VTpRMXTefgG4T8sw0RXHYx'

yelp_api = YelpAPI(g_MY_API)

g_perimeter = 100

#Get all nearby restaurants with a centroid(latitude, longitude) and the side of the square formed by the centroid
#Radius of the circle with the centroid = side / sqrt(2)
def get_nearby_restaurants(latitude, longitude, side):
  radius = round(side / math.sqrt(2))
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

# Calculate new longitude from old longitude
# increment = distance between new and old longitude (in meters)
# For this file, increment = perimeter
def get_new_longitude(old_longitude, increment):
  earth = 6378.137 #radius of the earth in kilometer
  pi = math.pi
  m = (1 / ((2 * pi / 360) * earth)) / 1000; #1 meter in degree
  new_longitude = old_longitude + (increment * m) / math.cos(old_longitude * pi / 180);
  return new_longitude

# Get all restaurants within a given perimeter for incrementing
# (centerline = the input latitude)
def get_restaurants_by_rows(min_longitude, max_longitude, lat, perimeter):
  restaurants_by_rows = []
  current_longitude = min_longitude
  while(current_longitude <= max_longitude):
    print(lat, ',', current_longitude)
    restaurants_by_rows.extend(get_nearby_restaurants(lat, current_longitude, perimeter))
    current_longitude = get_new_longitude(current_longitude, perimeter)
  restaurants_ids = [item['id'] for item in restaurants_by_rows]
  #remove duplicates and then return
  return [item for n, item in enumerate(restaurants_by_rows) if item not in restaurants_ids[n+1:]]
  #return restaurants_by_rows

#test
lat =  53.515096
g_MAX_LONG = 9.997166
g_MIN_LONG = 9.987242
results = get_restaurants_by_rows(g_MIN_LONG, g_MAX_LONG, lat, g_perimeter)

i = 0 #place breakpoint here to observe when debugging

# Calculate new latitude from old latitude
# increment = distance between new and old latitude (in meters)
# For this file, increment = perimeter
def get_new_latitude(old_latitude, increment):
  earth = 6378.137 #radius of the earth in kilometer
  pi = math.pi
  m = (1 / ((2 * pi / 360) * earth)) / 1000;  # 1 meter in degree
  new_latitude = old_latitude + (increment * m)
  return new_latitude

# Get all restaurants within a given perimeter for incrementing
# (centerline = the input longitude)
def get_restaurants_by_columns(min_latitude, max_latitude, longitude, perimeter):
  restaurants_by_columns = []
  current_latitude = min_latitude
  while(current_latitude <= max_latitude):
    print(current_latitude, ',', longitude)
    restaurants_by_columns.extend(get_nearby_restaurants(current_latitude, longitude, perimeter))
    current_latitude = get_new_latitude(current_latitude, perimeter)
  restaurants_ids = [item['id'] for item in restaurants_by_columns]
  # remove duplicates and then return
  return [item for n, item in enumerate(restaurants_by_columns) if item['id'] not in restaurants_ids[n + 1:]]
  #return restaurants_by_columns

g_MAX_LAT = 53.515681
g_MIN_LAT = 53.510501
g_long    = 9.987361

results = get_restaurants_by_columns(g_MIN_LAT, g_MAX_LAT, g_long, g_perimeter)

i = 0 #place breakpoint here to observe when debugging



# Get the restaurant density (restaurants/km) from a given centroid(latitude, longitude)
# side_in_km is the side of the square around the centroid
def get_density(latitude, longitude, side_in_km):
  restaurants = get_nearby_restaurants(latitude, longitude, side_in_km * 1000)
  print(restaurants)
  number_of_restaurants = len(restaurants)
  restaurants_per_km = number_of_restaurants / side_in_km
  return restaurants_per_km

#test
density = get_density(53.513216, 9.986797, 2)

i = 0 #place breakpoint here to observe when debugging
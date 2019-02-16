import json
import pandas as pd
import numpy as np
import pyproj
from collections import namedtuple

import os
os.chdir('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/JSON')

#definition of a restaurant
Restaurant = namedtuple("Restaurant",
                        'id \
                        epsg25832_latitude \
                        epsg25832_longitude \
                        epsg4326_latitude \
                        epsg4326_longitude \
                        price rating \
                        review_count \
                        is_closed \
                        zip_code \
                        category_restaurant_id \
                        category_alias \
                        category_title \
                        category_index'
                        )

with open('merged_extract.json') as json_data:
  businesses = json.load(json_data)

restaurants = [] #list of restaurants

for biz in businesses:
  try:
    price_temp = biz['price']
  except:
    price_temp = 'n/a'

  long_temp, lat_temp = transform_coord(biz['coordinates']['longitude'], biz['coordinates']['latitude'])

  restaurant = Restaurant(  #extract data into an instance of type Restaurant
    id = biz['id'],
    epsg4326_latitude = biz['coordinates']['latitude'],
    epsg4326_longitude = biz['coordinates']['longitude'],
    epsg25832_latitude = lat_temp,
    epsg25832_longitude = long_temp,
    price = price_temp,
    rating = biz['rating'],
    review_count = biz['review_count'],
    is_closed = biz['is_closed'],
    zip_code = biz['location']['zip_code'],
    category_restaurant_id = biz['id'],
    category_alias = biz['categories'][0]['alias'],
    category_title = biz['categories'][0]['title'],
    category_index = biz['id'] + biz['categories'][0]['alias']
  )
  restaurants.append(restaurant)

#print(Restaurant._fields)

raw_df_businesses = pd.DataFrame( #dataframe with all attributes of Restaurant
  data = restaurants,
  columns = Restaurant._fields
)

raw_head = raw_df_businesses.head()

df_businesses = raw_df_businesses[['id', 'price', 'rating', 'review_count', 'is_closed', 'zip_code',
                           'epsg25832_latitude', 'epsg25832_longitude',
                           'epsg4326_latitude', 'epsg4326_longitude']]

df_businesses.set_index('id', drop= True, inplace=True)

df_businesses = df_businesses[~df_businesses.index.duplicated(keep='first')]


df_businesses.to_csv('extract_businesses.csv')

df_categories = raw_df_businesses[['category_restaurant_id',
                                    'category_alias',
                                     'category_title']]
    
df_categories.set_index('category_restaurant_id', drop= True, inplace=True)

df_categories = df_categories[~df_categories.index.duplicated(keep='first')] 

df_categories.to_csv('extract_categories.csv')
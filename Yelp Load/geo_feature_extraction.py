import json
import pandas as pd
import numpy as np
import pyproj

def transform_coord(lat_in, long_in): #output to yelp
  inProj = pyproj.Proj(init='epsg:25832')
  outProj = pyproj.Proj(init='epsg:4326')
  lat_out,long_out = pyproj.transform(inProj,outProj,lat_in,long_in)
  return lat_out, long_out

with open('merged_extract.json') as json_data:
    d = json.load(json_data)
    json_data.close()
    
#businesses = d['businesses']
businesses = d


restaurant_id = []
epsg4326_latitude = []
epsg4326_longitude = []
epsg25832_latitude = []
epsg25832_longitude = []
price = []
rating = []
review_count = []
is_closed = []
zip_code = []
#df_categories = pd.DataFrame(columns = ['id', 'alias', 'title'])
i = 0

category_restaurant_id = []
category_alias = []
category_title = []
category_index = []

for biz in businesses:

    restaurant_id.append(biz['id'])
    #(Better do as list)category_alias = biz['categories'][0]['alias']
    #(Better do as list)category_title = biz['categories'][0]['title']
    epsg4326_latitude.append(biz['coordinates']['latitude'])
    epsg4326_longitude.append(biz['coordinates']['longitude'])
    lat_temp, long_temp = transform_coord(epsg4326_latitude, epsg4326_longitude)
    epsg25832_latitude.append(lat_temp)
    epsg25832_longitude.append(long_temp)
    try:
        price.append(biz['price'])
    except:
        price.append('n/a')
    rating.append(biz['rating'])
    review_count.append(biz['review_count'])
    is_closed.append(biz['is_closed'])
    zip_code.append(biz['location']['zip_code'])
    for cat in biz['categories']:
        category_restaurant_id.append(biz['id'])
        category_alias.append(cat['alias'])
        category_title.append(cat['title'])
        category_index.append(biz['id']+ '_' + cat['alias'])

df_businesses = pd.DataFrame(index=restaurant_id, data = {"rating": rating,
                                                               "review_count": review_count,
                                                               "is_closed": is_closed,
                                                               "epsg4326_latitude": epsg4326_latitude,
                                                               "epsg4326_longitude": epsg4326_longitude,
                                                               "epsg25832_latitude": epsg25832_latitude,
                                                               "epsg25832_longitude": epsg25832_longitude,                                                            
                                                               "zip_code": zip_code,
                                                               "price": price})

df_businesses = df_businesses[~df_businesses.index.duplicated(keep='first')]    

df_businesses.to_csv('extract_businesses.csv')



    

df_categories = pd.DataFrame(index = category_index, data = {'restaurant_id': category_restaurant_id,
                                     'alias': category_alias,
                                     'title': category_title})
    
df_categories = df_categories[~df_categories.index.duplicated(keep='first')] 

df_categories.to_csv('extract_categories.csv')


import json
import pandas as pd
import numpy as np

with open('merged_extract.json') as json_data:
    d = json.load(json_data)
    json_data.close()
    
#businesses = d['businesses']
businesses = d


restaurant_id = []
latitude = []
longitude = []
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
    latitude.append(biz['coordinates']['latitude'])
    longitude.append(biz['coordinates']['longitude'])
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
                                                               "latitude": latitude,
                                                               "longitude": longitude,
                                                               "zip_code": zip_code,
                                                               "price": price})

df_businesses = df_businesses[~df_businesses.index.duplicated(keep='first')]    

df_businesses.to_csv('extract_businesses.csv')



    

df_categories = pd.DataFrame(index = category_index, data = {'restaurant_id': category_restaurant_id,
                                     'alias': category_alias,
                                     'title': category_title})
    
df_categories = df_categories[~df_categories.index.duplicated(keep='first')] 

df_categories.to_csv('extract_categories.csv')

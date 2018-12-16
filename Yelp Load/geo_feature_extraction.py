import json
import pandas as pd
import numpy as np

with open('response.json') as json_data:
    d = json.load(json_data)
    json_data.close()
    
businesses = d['businesses']



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

for biz in businesses:

    restaurant_id.append(biz['id'])
    #(Better do as list)category_alias = biz['categories'][0]['alias']
    #(Better do as list)category_title = biz['categories'][0]['title']
    latitude.append(biz['coordinates']['latitude'])
    longitude.append(biz['coordinates']['longitude'])
    price.append(biz['price'])
    rating.append(biz['rating'])
    review_count.append(biz['review_count'])
    is_closed.append(biz['is_closed'])
    zip_code.append(biz['location']['zip_code'])
    for cat in biz['categories']:
        category_restaurant_id.append(biz['id'])
        category_alias.append(cat['alias'])
        category_title.append(cat['title'])

df_businesses = pd.DataFrame(index=restaurant_id, data = {"rating": rating,
                                                               "review_count": review_count,
                                                               "is_closed": is_closed,
                                                               "latitude": latitude,
                                                               "longitude": longitude,
                                                               "zip_code": zip_code,
                                                               "price": price})

df_categories = pd.DataFrame(data = {'restaurant_id': category_restaurant_id,
                                     'alias': category_alias,
                                     'title': category_title})


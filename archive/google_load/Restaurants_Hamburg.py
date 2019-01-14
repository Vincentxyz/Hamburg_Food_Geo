from googleplaces import GooglePlaces, types, lang
import pandas as pd
import decimal

API_KEY = -- Add API Key here

LOCATION = 'Hamburg, Germany'
RADIUS = 30000

google_places = GooglePlaces(API_KEY)

pagetoken = 'start'

ids= []
names = []
lats = []
lngs = []
tag_types = []
rating = []
reviews = []
count_reviews = []
vicinity = []

while len(pagetoken) != 0:
    
    if pagetoken == 'start':
    
        query_result = google_places.nearby_search(
                location=LOCATION,
                radius=RADIUS, types=[types.TYPE_RESTAURANT]
                )
    else:
    
        query_result = google_places.nearby_search(
                location=LOCATION,
                radius=RADIUS, 
                types=[types.TYPE_RESTAURANT],  
                pagetoken = pagetoken
                )
        
    
    for place in query_result.places:
        place.get_details()
        
        if (place.details.get('rating',1000)!=1000):
            ids.append(place.id)
            names.append(place.name)
            lats.append(place.geo_location['lat'])
            lngs.append(place.geo_location['lng'])
            tag_types.append(place.details['types'])
            rating.append(place.details['rating'])
            reviews.append(place.details['reviews'])
            count_reviews.append(len(place.details['reviews']))
            vicinity.append(place.details['vicinity'])
    
    pagetoken = query_result.next_page_token
    
    
    
df = pd.DataFrame({'id' : ids,
                  'name': names, 
                  'lat': lats,
                  'lng': lngs,
                  'types': tag_types,
                  'rating': rating,
                  'reviews': reviews,
                  'vicinity': vicinity})
    
df.to_json('Restaurants_Hamburg.json')

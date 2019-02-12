import pandas as pd
import math

github_path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/'

path = github_path + 'data/restaurants/'

restaurants = pd.read_csv(path + 'Restaurants_in_Hamburg.csv')

def euclidian_distance(x_old, x_new, y_old, y_new):
    return math.sqrt(math.pow(x_new-x_old,2) + math.pow(y_new-y_old,2))

def get_number_of_nearby_restaurants(x, y, radius):
    no_restaurants = 0
    for i in range(0,len(restaurants)):
        x_dest = restaurants.iloc[i,6]
        y_dest = restaurants.iloc[i,7]
        if euclidian_distance(x, x_dest, y, y_dest) <= radius:
            no_restaurants += 1
            
    return no_restaurants




restaurant_densities = []
restaurant_ids = []

for i in range(0,len(restaurants)):        
    restaurant_densities.append(get_number_of_nearby_restaurants(restaurants.iloc[i,6],restaurants.iloc[i,7],400))
    restaurant_ids.append(restaurants.iloc[i,0])

densities = pd.DataFrame({'id': restaurant_ids, 'density': restaurant_densities})

densities.to_csv(github_path + '/data/restaurant_densities/restaurant_densities.csv')

import matplotlib.pyplot as plt
plt.hist(restaurant_densities)
        
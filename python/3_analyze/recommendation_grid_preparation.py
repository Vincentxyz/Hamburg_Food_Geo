import pandas as pd
github_path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo'

### Calculating the recommendation grid ######
import math
path = github_path + '/data/restaurants/'

restaurants = pd.read_csv(path + 'Restaurants_in_Hamburg.csv')

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
    return distance

def get_number_of_nearby_restaurants(x, y, radius):
    no_restaurants = 0
    for i in range(0,len(restaurants)):
        x_dest = restaurants.iloc[i,6]
        y_dest = restaurants.iloc[i,7]
        if calculate_distance(x, y, x_dest, y_dest) <= radius:
            no_restaurants += 1
            
    return no_restaurants

water_points = pd.read_csv(github_path + '/data/water_distances/water_points.csv', sep = ';')


MAX_x_25832 = 588050
MAX_y_25832 = 5955199

MIN_x_25832 = 548145
MIN_y_25832 = 5916563


TILE_SIZE = 1000

x, y = MIN_x_25832 + TILE_SIZE/2, MIN_y_25832 + TILE_SIZE/2


x_values = []
y_values = []
distance_to_water = []
restaurant_densities = []

while (x < (MAX_x_25832 - (TILE_SIZE/2))):
  y = MIN_y_25832
  while (y < (MAX_y_25832 - (TILE_SIZE/2))):
    min_distance = 99999999999.0
#    for i in range(len(water_points)):
#        distance = calculate_distance(x,
#                                      y,
#                                      water_points.iloc[i,1],
#                                      water_points.iloc[i,2])
#        
#        if distance < min_distance:
#            min_distance = distance
#    
#    distance_to_water.append(min_distance)
#    
#    restaurant_densities.append(get_number_of_nearby_restaurants(x,y,400))
#    
    x_values.append(x)
    y_values.append(y)
    print('point ' + str(x) + ', ' + str(y) + ' finished')
      
    y = y + TILE_SIZE
    
  x = x + TILE_SIZE


recommendation_centroids = pd.DataFrame({'x': x_values, 'y': y_values, 'distance_to_water': distance_to_water,
                                         'restaurant_density': restaurant_densities})

recommendation_centroids.to_csv(github_path + '/data/recommendation_grid/water_distance_and_density.csv')



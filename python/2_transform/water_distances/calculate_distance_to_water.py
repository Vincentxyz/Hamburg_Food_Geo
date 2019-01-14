import pandas as pd
import numpy as np
import math

df_restaurants = pd.read_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/Restaurants_in_Hamburg.csv')
df_restaurants.set_index('id', inplace = True, drop = True)

water_points = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/QGIS_Projects/geokarte_grid/water_points.csv', sep = ';')
#water_points = water_points[:,1:3]

distance_to_water = []
ids = []

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt(math.pow(x2-x1,2) + math.pow(y2-y1,2))
    return distance

#for i in range(582,len(df_restaurants)):
for i in range(582,586):
    min_distance = 99999999999.0
    
    for j in range(len(water_points)):
        distance = calculate_distance(df_restaurants.iloc[i,5],
                                      df_restaurants.iloc[i,6],
                                      water_points.iloc[j,1],
                                      water_points.iloc[j,2])
        
        if distance < min_distance:
            min_distance = distance
    
    distance_to_water.append(min_distance)
    ids.append(df_restaurants.index[i])
    print('restaurant ' + str(i) + ' finished')
    
distances = pd.DataFrame({'distance': distance_to_water}, index = ids)
distances.to_csv('distances_' + str(583) + '_' + str(i+1) + '.csv')
df_restaurants['water_distance'] = distance_to_water

df_part_restaurants = df_restaurants.iloc[0:2496,:]
df_part_restaurants['water_distance'] = distance_to_water
df_part_restaurants.to_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/Restaurants_in_Hamburg_with_water_distance.csv')

df_restaurants.to_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/Restaurants_in_Hamburg_with_water_distance.csv')



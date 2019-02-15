import pandas as pd

water_distance_and_density = pd.read_csv('water_distance_and_density.csv')

new_x_y = water_distance_and_density.iloc[:,1:3].drop_duplicates()


grid_centroids_with_district = pd.read_csv('grid_centroids_with_district.csv')

new_x_y = grid_centroids_with_district.iloc[:,1:3].drop_duplicates()
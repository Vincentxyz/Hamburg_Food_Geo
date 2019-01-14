import pandas as pd

geo_grid = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/QGIS_Projects/geokarte_grid/geokarte_grid_tiny.xyz', sep = ' ', header = None, names = ['lat', 'long', 'illumination'])

water_lat = []
water_long = []

for i in range(len(geo_grid)):
#for i in range(10):
    if geo_grid.iloc[i,2] == 0:
        water_lat.append(geo_grid.iloc[i,0])
        water_long.append(geo_grid.iloc[i,1])
        
water_points = pd.DataFrame({'lat': water_lat, 'long': water_long})

water_points.to_csv('C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/QGIS_Projects/geokarte_grid/water_points.csv', sep = ';')

import pyproj
import numpy as np
from osgeo import gdal

lat1 = 548145
lon1 = 5916565

lon2 = 2
lat2 = 49

p1 = pyproj.Proj(init='epsg:25832')

#WGS 84
p2 = pyproj.Proj(init='epsg:4326')



x1, y1 = pyproj.transform(p1, p2, lat1, lon1)

print(x1, y1)

x2, y2 = pyproj.transform(epsg25832, wgs84, lon2, lat2)
print(x2, y2)

# a Pythagore's theorem is sufficient to compute an approximate distance
distance_m = np.sqrt((x2-x1)**2 + (y2-y1)**2)
print(distance_m)
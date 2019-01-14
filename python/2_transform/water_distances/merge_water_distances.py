import pandas as pd

path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/data/water_distances/'
df_distances_1_582 = pd.read_csv(path + 'distances_1_582.csv')
df_distances_582_1455 = pd.read_csv(path + 'distances_582_1455.csv')
df_distances_1456_2495 = pd.read_csv(path + 'distances_1456_2495.csv')
df_distances_1456_2495 = df_distances_1456_2495.drop('no',axis=1)
df_distances_2496_3909 = pd.read_csv(path + 'distances_2496_3909.csv')

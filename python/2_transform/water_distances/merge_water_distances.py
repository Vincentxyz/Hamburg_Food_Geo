import pandas as pd




# Read in distances files
path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/data/water_distances/'
#
#df_distances_582_1455 = pd.read_csv(path + 'distances_582_1455.csv')
#df_distances_583_1455 = df_distances_582_1455.iloc[1:,:]
#df_distances_583_1455.to_csv(path  + 'distances_583_1455.csv')

df_distances_1_582 = pd.read_csv(path + 'distances_1_582.csv')
df_distances_583_1455 = pd.read_csv(path + 'distances_583_1455.csv')
df_distances_1456_2495 = pd.read_csv(path + 'distances_1456_2495.csv')
df_distances_2496_3909 = pd.read_csv(path + 'distances_2496_3909.csv')

# Drop unnecessary columns
df_distances_1456_2495 = df_distances_1456_2495.drop('no',axis=1)
df_distances_583_1455 = df_distances_583_1455.drop('no',axis=1)
df_distances_583_1455 = df_distances_583_1455.drop('Unnamed: 0',axis=1)
df_distances_583_1455 = df_distances_583_1455.drop('Unnamed: 0.1',axis=1)
df_distances_2496_3909 = df_distances_2496_3909.drop('no',axis=1)


# Set indexes
df_distances_1_582.set_index('Unnamed: 0', drop = True, inplace= True)
df_distances_583_1455.set_index('id', drop = True, inplace= True)
df_distances_1456_2495.set_index('Unnamed: 0', drop = True, inplace= True)
df_distances_2496_3909.set_index('Unnamed: 0', drop = True, inplace= True)


df_distances = df_distances_1_582.append(other = df_distances_583_1455,
                        verify_integrity = True
                        )



df_distances = df_distances.append(other = df_distances_1456_2495,
                        verify_integrity = True
                        )

df_distances = df_distances.append(other = df_distances_2496_3909,
                        verify_integrity = True
                        )

df_distances.to_csv(path + 'all_water_distances.csv')
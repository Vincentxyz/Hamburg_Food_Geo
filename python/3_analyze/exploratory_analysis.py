import pandas as pd
import numpy as np
from sklearn import preprocessing

github_path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo'

df_restaurants = pd.read_csv(github_path + '/data/restaurants/Restaurants_in_Hamburg.csv')
df_social_values = pd.read_csv(github_path + '/data/social_values/Social_Values_2016_adjusted.csv')
df_water_distances = pd.read_csv(github_path + '/data/water_distances/all_water_distances.csv')
df_water_distances = df_water_distances.rename(columns = {'Unnamed: 0' : 'id',
                                                          'distance' : 'distance_to_water'})
df_crimes = pd.read_csv(github_path + '/data/crime_rate/crime_rates.csv')

df_restaurant_densities = pd.read_csv(github_path + '/data/restaurant_densities/restaurant_densities.csv')

df_all = pd.merge(left=df_restaurants, right=df_social_values,
                left_on='Stadtteil', right_on = 'Stadtgebiet')

df_all = pd.merge(left = df_all, right = df_water_distances,
                  left_on = 'id', right_on = 'id') 

df_all = pd.merge(left = df_all, right = df_restaurant_densities,
                  left_on = 'id', right_on = 'id')

df_all = pd.merge(left = df_all, right = df_crimes,
                  left_on = 'Stadtteil', right_on='City_part')


scaler = preprocessing.StandardScaler()

normalized_rating = scaler.fit_transform(np.array(df_all['rating']).reshape(-1,1))

normalized_review_count =scaler.fit_transform(np.array(df_all['review_cou']).reshape(-1,1))
    
df_all['success'] = normalized_rating + normalized_review_count


# Add categories

df_categories = pd.read_csv(github_path + '/data/restaurant_categories/extract_categories.csv')
df_categories = df_categories.iloc[:,0:len(df_categories.columns)-1]

df_categories_wo_restaurant = df_categories.iloc[:, 1]
# Encoding the categories
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_category = LabelEncoder()
df_categories_wo_restaurant = labelencoder_category.fit_transform(df_categories_wo_restaurant)
onehotencoder = OneHotEncoder(categorical_features = [0])
df_categories_wo_restaurant = onehotencoder.fit_transform(df_categories_wo_restaurant.reshape(-1,1)).toarray()

df_categories = pd.concat(objs = [df_categories.iloc[:,0],pd.DataFrame(df_categories_wo_restaurant)], axis = 1)

df_category_grouped_by = df_categories.groupby('category_restaurant_id').sum()


#businesses['success'] = succes

df_features = pd.DataFrame({
        'id': df_all['id'],
        'price' : df_all['price'].fillna('unknown'),
        
        # Water distance
        'distance_to_water' : df_all['distance_to_water'],
        
        #Restaurant density
        'restaurant_density': df_all['density'],
        
        #Crimes
        'crime_count': df_all['Criminal_cases'],
        
        #social values
        'population': df_all['Bevölkerung'],
        'share_under_18_years_old': df_all['Unter 18-Jährige'],
        #'65_year_olds_and_older': df_all[' 65-Jährige und Ältere'],
        'share_65_year_olds_and_older': df_all['Anteil der 65-Jährigen und Älteren in %'],
        #'foreigners': df_all['Ausländerinnen und Ausländer'],
        'share_foreigners': df_all['Ausländeranteil in %'],
        'population_with_migration_background': df_all['Bevölkerung mit Migrations-hintergrund'],
        'share_population_with_migration_background': df_all['Anteil der Bevölkerung mit Migrations-hintergrund in %'],
        #'below_18_year_olds_with_migration_background' : df_all['Unter 18-Jährige mit Migrations-hintergrund'],
        #'share_below_18_year_olds_with_migration_background': df_all['Anteil der unter 18-Jährigen mit Migrations-hintergrund in %'],
        'households' : df_all['Haushalte'],
        #'people_per_household' : df_all['Personen je Haushalt'],
        #'one_person_households' : df_all['Einpersonen-haushalte'],
        'share_one_person_households' : df_all['Anteil der Einpersonen-haushalte in %'],
        #'households_with_children' : df_all['Haushalte mit Kindern'],
        'share_households_with_children': df_all['Anteil der Haushalte mit Kindern in %'],
        #'single_parents' : df_all['Alleinerziehende'],
        'share_households_with_single_parents' : df_all['Anteil der Haushalte von Alleinerziehenden in %'],
        #'area_in_square_kms' : df_all['Fläche in km²'],
        'population_density' : df_all['Bevölkerungs-dichte'],
        #'births': df_all['Geburten'],
        #'deaths': df_all['Sterbefälle'],
        #'immigration': df_all['Zuzüge'],
        #'emigration': df_all['Fortzüge'],
        #'migration_balance': df_all['Wanderungssaldo'],
        #'insurable_employees': df_all['Sozial-versicherungs-pflichtig Beschäftigte (Dez 2016)'],
        'employment_quote_in_%': df_all['Beschäftigten-quote in % (Dez 2016)'],
        #'unemployed_people': df_all['Arbeitslose (Dez 2016)'],
        'share_unemployed_people': df_all['Arbeitslosenanteil in % (Dez 2016)'],
        #'younger_unemployed_people' : df_all['Jüngere Arbeitslose (Dez 2016)'],
        #'share_younger_unemployed_people': df_all['Arbeitslosenanteil Jüngerer in % (Dez 2016)'],
        #'older_unemployed_people': df_all['Ältere Arbeitslose (Dez 2016)'],
        #'share_older_unemployed_people': df_all['Arbeitslosenanteil Älterer in % (Dez 2016)'],
        #'service_recipients_per_SGB_II' : df_all['Leistungs-empfänger/-innen nach SGB II (Dez 2016)'],
        #'share_service_recipients_per_SGB_II': df_all['Anteil der Leistungs-empfänger/-innen nach SGB II in % (Dez 2016)'],
        #'under_15_year_olds_in_minimum_income': df_all['Unter 15-Jährige in Mindestsicherung (Dez 2016)'],
        #'share_under_15_year_olds_in_minimum_income': df_all['Anteil der unter 15-Jährigen in Mindestsicherung in % (Dez 2016)'],
        #'shared_households_per_SGB_II': df_all['Bedarfs-gemeinschaften nach SGB II (Dez 2016)'],
        #'tax_liable_people' : df_all['Lohn- und Einkommen-steuerpflichtige (2013)'],
        'sum_of_incomes_per_tax_reliable_person_in_EUR': df_all['Gesamtbetrag der Einkünfte je Steuerpflichtigen in EUR (2013)'],
        #'residential_buildings' : df_all['Wohngebäude (2016)'],
        #'apartments' : df_all['Wohnungen (2016)'],
        #'apartments_ready_for_occupancy' : df_all['Bezugsfertige Wohnungen (2016)'],
        #'homes_in_one_or_two_family_houses': df_all['Wohnungen in Ein- und Zweifamilien-häusern (2016)'],
        #'share_homes_in_one_or_two_family_houses': df_all['Anteil der Wohnungen in Ein- und Zweifamilien-häusern in % (2016)'],
        #'home_size_in_sqm': df_all['Wohnungsgröße in m² (2016)'], 
        #'living_area_per_resident_in_sqm' : df_all['Wohnfläche je Einwohner/-in in m² (2016)'],
        #'social_homes': df_all['Sozialwohnungen (Jan 2017)'],
        #'share_social_homes' : df_all['Sozialwohnungs-anteil in % (Jan 2017)'],
        #'social_homes_with_ends_of_commitment_until_2022' : df_all['Sozialwohnungen mit Bindungsauslauf bis 2022'],
        #'share_social_homes_with_ends_of_commitment_until_2022': df_all['Sozialwohnungen mit Bindungsauslauf bis 2022 in %'],
        'prices_for_properties' : df_all['Preise für Grundstücke in EUR/m² (Jan 2017)'],
        #'prices_for_one_or_two_family_houses_in_EUR/sqm' : df_all['Preise für Ein- bzw Zwei-familienhäuser in EUR/m² (Jan 2017)'],
        'prices_for_condominiums' : df_all['Preise für Eigentums-wohnungen in EUR/m² (Jan 2017)'],
        #'kindergardens_and_preschool_classes': df_all['Kindergärten und Vorschulklassen (März 2017)'],
        #'primary_schools': df_all['Grundschulen (2016/2017)'],
        #'students_in_secondary_level_1': df_all['Schülerinnen und Schüler der Sekundarstufe I (2016/2017)'],
        #'share_students_in_city_district_schools' : df_all['Anteil der Schülerinnen und Schüler in Stadtteilschulen in % (2016/2017)'],
        'share_students_in_Gymnasium' : df_all['Anteil der Schülerinnen und Schüler in Gymnasien in % (2016/2017)'],
        #'residential_practitioners': df_all['Niedergelassene Ärzte (Jan 2017)'],
        #'general_practitioners': df_all['Allgemeinärzte (Jan 2017)'],
        #'dentists': df_all['Zahnärzte (Dez 2016)'],
        #'pharmacys': df_all['Apotheken (Dez 2016)'],
        #'private_cars': df_all['Private PKW (Jan 2017)'],
        'car_density' : df_all['PKW-Dichte (Jan 2017)']
    
  })

df_features = pd.merge(left = df_features, right = df_categories,
                       left_on = 'id', right_on = 'category_restaurant_id')

# For Sachin's Excel
#df_features.to_csv('restaurant_features.csv',encoding='utf-8')

df_features.set_index('id', inplace=True,drop=True)


from sklearn.preprocessing import LabelEncoder

# label encode price
labelencoder_price = LabelEncoder()
labelencoder_price = labelencoder_price.fit(['unknown', '€','€€','€€€','€€€€','€€€€€'])
df_features['price'] = labelencoder_price.transform(df_features['price'])

# Replace strange values

df_features['prices_for_one_or_two_family_houses_in_EUR/sqm'] = pd.to_numeric('prices_for_one_or_two_family_houses_in_EUR/sqm', errors = 'coerce')

for i in range(len(df_features.columns)):
    feature_column = df_features.iloc[:,i]
    feature_column = pd.to_numeric(feature_column, errors='coerce')
    df_features[df_features.columns[i]] = feature_column

# Fill missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
df_features = imputer.fit_transform(df_features)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_features, df_all['success'], test_size = 0.3,random_state = 0)

# scale numerical values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0, max_depth = 5)
rf_regressor.fit(X_train, y_train)

# Predicting the Test set results
rf_y_pred_train = rf_regressor.predict(X_train)
rf_y_pred_test = rf_regressor.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print("R-squared for training data: " + str(r2_score(y_train, rf_y_pred_train)))
print("R-squared for test data: " + str(r2_score(y_test, rf_y_pred_test)))




######Mulilinear Regression#######
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
l_regressor = LinearRegression()
l_regressor.fit(X_train, y_train)

# Predicting the Test set results
l_y_pred_train = l_regressor.predict(X_train)
l_y_pred_test = l_regressor.predict(X_test)

print("R-squared for training data: " + str(r2_score(y_train, l_y_pred_train)))
print("R-squared for test data: " + str(r2_score(y_test, l_y_pred_test)))






### Calculating the recommendation grid ######
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

water_points = pd.read_csv('C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/QGIS_Projects/geokarte_grid/water_points.csv', sep = ';')


MAX_x_25832 = 588050
MAX_y_25832 = 5955199

MIN_x_25832 = 548145
MIN_x_25832 = 5916563


TILE_SIZE = 1000

x, y = MIN_x_25832 + TILE_SIZE/2, MIN_y_25832 + TILE_SIZE/2


x_values = []
y_values = []
distance_to_water = []
restaurant_densities = []

while (x < (MAX_x_25832 - (TILE_SIZE/2))):
  y = MIN_LONG_25832
  while (y < (MAX_y_25832 - (TILE_SIZE/2))):
    for i in range(len(water_points)):
        distance = calculate_distance(x,
                                      y,
                                      water_points.iloc[i,1],
                                      water_points.iloc[i,2])
        
        if distance < min_distance:
            min_distance = distance
    
    distance_to_water.append(min_distance)
    
    restaurant_densities.append(get_number_of_nearby_restaurants(x,y,400))
    
    x_values.append(x)
    y_values.append(y)
    print('point ' + str(x) + ', ' + str(y) + ' finished')
      
    y = y + TILE_SIZE
    
  x = x + TILE_SIZE


recommendation_centroids = pd.DataFrame({'x': x_values, 'y': y_values, 'distance_to_water': distance_to_water,
                                         'restaurant_density': restaurant_densities})

recommendation_centroids.to_csv(github_path + '/data/recommendation_grid/water_distance_and_density.csv')





#################### Exploratory Data Analysis ########################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random



plt.scatter(x= "distance_to_water", y= "success", data = df_all.sample(200))

plt.scatter(x= "Bevölkerungs-dichte", y= "success", data = df_all)

plt.scatter(x= "Gesamtbetrag der Einkünfte je Steuerpflichtigen in EUR (2013)", y= "success", data = df_all)


sns.



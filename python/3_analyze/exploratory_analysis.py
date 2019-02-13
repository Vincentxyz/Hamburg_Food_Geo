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
                left_on='Stadtteil', right_on = 'city_part')

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
        'population': df_all['population'],
        'share_under_18_years_old': df_all['share_under_18_years_old'],
        'share_65_year_olds_and_older': df_all['share_65_year_olds_and_older'],
        'share_foreigners': df_all['share_foreigners'],
        'population_with_migration_background': df_all['population_with_migration_background'],
        'share_population_with_migration_background': df_all['share_population_with_migration_background'],
        'households' : df_all['households'],
        'share_one_person_households' : df_all['share_one_person_households'],
        'share_households_with_children': df_all['share_households_with_children'],
        'share_households_with_single_parents' : df_all['share_households_with_single_parents'],
        'population_density' : df_all['population_density'],
        'employment_quote_in_%': df_all['employment_quote_in_%'],
        'share_unemployed_people': df_all['share_unemployed_people'],
        'sum_of_incomes_per_tax_reliable_person_in_EUR': df_all['sum_of_incomes_per_tax_reliable_person_in_EUR'],
        'prices_for_properties' : df_all['prices_for_properties'],
        'prices_for_condominiums' : df_all['prices_for_condominiums'],
        'share_students_in_Gymnasium' : df_all['share_students_in_Gymnasium'],
        'car_density' : df_all['car_density']
    
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




################## Build Recommendation Grid #######
recommendation_centroids = pd.read_csv(github_path + '/data/recommendation_grid/grid_centroids_with_district.csv')

recommendation_centroids = recommendation_centroids.iloc[:,1:9]

recommendation_centroids = recommendation_centroids.drop(['gml_id','OBJECTID','Bezirk_Name'],axis=1)

centroid_social_values = pd.merge(left=recommendation_centroids, right=df_social_values,
                left_on='Stadtteil', right_on = 'city_part')

recommendation_centroids = pd.merge(left = recommendation_centroids, right = df_crimes,
                  left_on = 'Stadtteil', right_on='City_part')


# Now build feature vector


#################### Exploratory Data Analysis ########################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random



plt.scatter(x= "distance_to_water", y= "success", data = df_all.sample(200))

plt.scatter(x= "Bevölkerungs-dichte", y= "success", data = df_all)

plt.scatter(x= "Gesamtbetrag der Einkünfte je Steuerpflichtigen in EUR (2013)", y= "success", data = df_all)


sns.



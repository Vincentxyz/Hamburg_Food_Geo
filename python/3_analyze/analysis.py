import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

github_path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo'

def calc_adjusted_r_squared(r_squared, n, p):
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n-1) / (n-p-1)
    
    return adj_r_squared

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


# Add categories

df_categories = pd.read_csv(github_path + '/data/restaurant_categories/extract_categories.csv')
df_categories = df_categories.iloc[:,0:len(df_categories.columns)-1]

df_categories_wo_restaurant = df_categories.iloc[:, 1]
# Encoding the categories
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_category = LabelEncoder()
labelencoder_category = labelencoder_category.fit(df_categories_wo_restaurant)
df_categories_wo_restaurant = labelencoder_category.transform(df_categories_wo_restaurant)
onehotencoder = OneHotEncoder(categorical_features = [0])
df_categories_wo_restaurant = onehotencoder.fit_transform(df_categories_wo_restaurant.reshape(-1,1)).toarray()

df_categories_new = pd.DataFrame(df_categories_wo_restaurant)
df_categories_new['category_restaurant_id'] = df_categories['category_restaurant_id']

df_all = pd.merge(left = df_all, right = df_categories_new,
                       left_on = 'id', right_on = 'category_restaurant_id')


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
        'car_density' : df_all['car_density'],
    
  })


df_features = pd.merge(left = df_features, right = df_categories_new,
                       left_on = 'id', right_on = 'category_restaurant_id')

df_features = df_features.drop(['category_restaurant_id','id'],axis = 1)

from sklearn.preprocessing import LabelEncoder

# label encode price
labelencoder_price = LabelEncoder()
labelencoder_price = labelencoder_price.fit(['unknown', '€','€€','€€€','€€€€','€€€€€'])
df_features['price'] = labelencoder_price.transform(df_features['price'])

# Replace strange values

for i in range(len(df_features.columns)):
    feature_column = df_features.iloc[:,i]
    feature_column = pd.to_numeric(feature_column, errors='coerce')
    df_features[df_features.columns[i]] = feature_column

feature_columns = df_features.columns.values
distinct_categories = df_categories['category_alias'].unique()
feature_columns[(len(feature_columns)-len(distinct_categories)):(len(feature_columns))] = labelencoder_category.inverse_transform(pd.to_numeric(feature_columns[(len(feature_columns)-len(distinct_categories)):(len(feature_columns))]))

# Fill missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
df_features = imputer.fit_transform(df_features)

# scale numerical values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
df_features = sc_X.fit_transform(df_features)




####### Random Forest Regression #######
rf_training_r2 = []
rf_test_r2 = []
svr_training_r2 = []
svr_test_r2 = []
lr_training_r2 = []
lr_test_r2 = []

rf_training_adj_r2 = []
rf_test_adj_r2 = []
svr_training_adj_r2 = []
svr_test_adj_r2 = []
lr_training_adj_r2 = []
lr_test_adj_r2 = []



# Change no_fittings to 100 for an analysis of the different regressors, 1 for using the regressor for the recommendations 
no_fittings = 100

for i in range(0,no_fittings):
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_all['success'], test_size = 0.3,random_state = 0)

    
    # Fitting Random Forest Regression to the Training set
    from sklearn.ensemble import RandomForestRegressor
    rf_regressor = RandomForestRegressor(n_estimators = 1000, max_depth = 5)
    rf_regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    rf_y_pred_train = rf_regressor.predict(X_train)
    rf_y_pred_test = rf_regressor.predict(X_test)
    
    from sklearn.metrics import r2_score, mean_squared_error
    print("Random Forest: R-squared for training data: " + str(r2_score(y_train, rf_y_pred_train)))
    print("Random Forest: R-squared for test data: " + str(r2_score(y_test, rf_y_pred_test)))
    print("Random Forest: Adjusted R-squared for training data"+ str(calc_adjusted_r_squared(r2_score(y_train, rf_y_pred_train),len(pd.DataFrame(df_features).columns),1)))
    print("Random Forest: Adjusted R-squared for test data"+ str(calc_adjusted_r_squared(r2_score(y_test, rf_y_pred_test),len(pd.DataFrame(df_features).columns),1)))

    
    rf_training_r2.append(r2_score(y_train, rf_y_pred_train))
    rf_test_r2.append(r2_score(y_test, rf_y_pred_test))
    rf_training_adj_r2.append(calc_adjusted_r_squared(r2_score(y_train, rf_y_pred_train),len(pd.DataFrame(df_features).columns),1))
    rf_test_adj_r2.append(calc_adjusted_r_squared(r2_score(y_test, rf_y_pred_test),len(pd.DataFrame(df_features).columns),1))
    
    rf_feature_importances = pd.concat([pd.DataFrame(feature_columns),pd.DataFrame(rf_regressor.feature_importances_)],axis=1)
    rf_feature_importances.columns = ['attribute','importance']
    rf_feature_importances = rf_feature_importances.sort_values(by = ['importance'],ascending = False)
    rf_feature_importances.to_csv(github_path + '/data/feature_importances/rf_feature_importances.csv')
    
    rf_feature_importances['importance'] = rf_feature_importances['importance'] * 100
#    
#    ax_feature_importance = sns.barplot(x= "importance", y= "attribute", 
#                data = rf_feature_importances.iloc[0:10,:], orient = 'h', 
#                color = 'blue')
#    
#    ax_feature_importance.set(xlabel='Feature Importance in %', ylabel='Feature')

    
    ####### Support Vector Regression ########
    
    # Fitting SVR to the dataset
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    
    # Predicting a new result
    svr_y_pred_train = regressor.predict(X_train)
    svr_y_pred_test = regressor.predict(X_test)
    
    print("SVR: R-squared for training data: " + str(r2_score(y_train, svr_y_pred_train)))
    print("SVR: R-squared for test data: " + str(r2_score(y_test, svr_y_pred_test)))
    print("SVR: Adjusted R-squared for training data"+ str(calc_adjusted_r_squared(r2_score(y_train, svr_y_pred_train),len(pd.DataFrame(df_features).columns),1)))
    print("SVR: Adjusted R-squared for test data"+ str(calc_adjusted_r_squared(r2_score(y_test, svr_y_pred_test),len(pd.DataFrame(df_features).columns),1)))

    
    svr_test_r2.append(r2_score(y_test, svr_y_pred_test))
    svr_training_r2.append(r2_score(y_train, svr_y_pred_train))
    svr_training_adj_r2.append(calc_adjusted_r_squared(r2_score(y_train, svr_y_pred_train),len(pd.DataFrame(df_features).columns),1))
    svr_test_adj_r2.append(calc_adjusted_r_squared(r2_score(y_test, svr_y_pred_test),len(pd.DataFrame(df_features).columns),1))
    
    ###### Mulivariate linear Regressio #######
    # Fitting Multiple Linear Regression to the Training set
    
    from sklearn.linear_model import LinearRegression
    lr_regressor = LinearRegression()
    lr_regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    lr_y_pred_train = lr_regressor.predict(X_train)
    lr_y_pred_test = lr_regressor.predict(X_test)
    
    print("Linear Regression: R-squared for training data: " + str(r2_score(y_train, lr_y_pred_train)))
    print("Linear Regression: R-squared for test data: " + str(r2_score(y_test, lr_y_pred_test)))

    
    lr_training_r2.append(r2_score(y_train, lr_y_pred_train))
    lr_test_r2.append(r2_score(y_test, lr_y_pred_test))
       
    lr_training_adj_r2.append(calc_adjusted_r_squared(r2_score(y_train, lr_y_pred_train),len(pd.DataFrame(df_features).columns),1))
    lr_test_adj_r2.append(calc_adjusted_r_squared(r2_score(y_test, lr_y_pred_test),len(pd.DataFrame(df_features).columns),1))

# Save training r2 values

df_training_rf_r2 = pd.DataFrame({'Regressor' : 'RFR', 'R2' : rf_training_r2})
df_training_svr_r2 = pd.DataFrame({'Regressor' : 'SVR', 'R2' : svr_training_r2})
df_training_lr_r2 = pd.DataFrame({'Regressor' : 'LR', 'R2' : lr_training_r2})

df_r2_training = df_training_rf_r2.append(df_training_svr_r2, ignore_index=True)
df_r2_training = df_r2_training.append(df_training_lr_r2, ignore_index = True)

#df_r2_training.to_csv(github_path + '/data/regression_comparison/r2_training_data.csv')

sns.boxplot(x = 'Regressor', y = 'R2', data = df_r2_training)

# Save training adj_r2 values

df_training_rf_adj_r2 = pd.DataFrame({'Regressor' : 'RFR', 'Adjusted R2' : rf_training_adj_r2})
df_training_svr_adj_r2 = pd.DataFrame({'Regressor' : 'SVR', 'Adjusted R2' : svr_training_adj_r2})
df_training_lr_adj_r2 = pd.DataFrame({'Regressor' : 'LR', 'Adjusted R2' : lr_training_adj_r2})

df_adj_r2_training = df_training_rf_adj_r2.append(df_training_svr_adj_r2, ignore_index=True)
df_adj_r2_training = df_adj_r2_training.append(df_training_lr_adj_r2, ignore_index = True)

#df_adj_r2_training.to_csv(github_path + '/data/regression_comparison/r2_training_data.csv')


# With linear legression
sns.boxplot(x = 'Regressor', y = 'Adjusted R2', data = df_adj_r2_training)


# Without linear legression
sns.boxplot(x = 'Regressor', y = 'Adjusted R2', data = df_adj_r2_training[df_adj_r2_training['Regressor'] != 'LR'])

# Save test r2 values

df_test_rf_r2 = pd.DataFrame({'Regressor' : 'RFR', 'R2' : rf_test_r2})
df_test_svr_r2 = pd.DataFrame({'Regressor' : 'SVR', 'R2' : svr_test_r2})
df_test_lr_r2 = pd.DataFrame({'Regressor' : 'LR', 'R2' : lr_test_r2})

df_r2_test = df_test_rf_r2.append(df_test_svr_r2, ignore_index=True)
df_r2_test = df_r2_test.append(df_test_lr_r2, ignore_index = True)

#df_r2_test.to_csv(github_path + '/data/regression_comparison/r2_test_data.csv')

# With linear regression
sns.boxplot(x = 'Regressor', y = 'R2', data = df_r2_test)

# Without linear legression
sns.boxplot(x = 'Regressor', y = 'R2', data = df_r2_test[df_r2_test['Regressor'] != 'LR'])


# Save test adj_r2 values

df_test_rf_adj_r2 = pd.DataFrame({'Regressor' : 'RFR', 'Adjusted R2' : rf_test_adj_r2})
df_test_svr_adj_r2 = pd.DataFrame({'Regressor' : 'SVR', 'Adjusted R2' : svr_test_adj_r2})
df_test_lr_adj_r2 = pd.DataFrame({'Regressor' : 'LR', 'Adjusted R2' : lr_test_adj_r2})

df_adj_r2_test = df_test_rf_adj_r2.append(df_test_svr_adj_r2, ignore_index=True)
df_adj_r2_test = df_adj_r2_test.append(df_test_lr_adj_r2, ignore_index = True)

#df_adj_r2_test.to_csv(github_path + '/data/regression_comparison/r2_test_data.csv')

# With linear legression
sns.boxplot(x = 'Regressor', y = 'Adjusted R2', data = df_adj_r2_test)


# Without linear legression
sns.boxplot(x = 'Regressor', y = 'Adjusted R2', data = df_adj_r2_test[df_adj_r2_training['Regressor'] != 'LR'])


################## Build Recommendation Grid #######
recommendation_centroids = pd.read_csv(github_path + '/data/recommendation_grid/grid_centroids_with_district.csv')

recommendation_centroids = recommendation_centroids.iloc[:,1:9]

recommendation_centroids = recommendation_centroids.drop(['gml_id','OBJECTID','Bezirk_Name'],axis=1)

recommendation_centroids = pd.merge(left = recommendation_centroids, right = df_crimes,
                  left_on = 'Stadtteil', right_on='City_part',how='left')

centroid_social_values = pd.merge(left=recommendation_centroids, right=df_social_values,
                left_on='Stadtteil', right_on = 'city_part',how='left')




# Now build feature vector

price_levels = ['€','€€','€€€','€€€€','€€€€€']
categories = df_categories['category_alias'].unique()

frequent_categories = df_categories.groupby(by=['category_alias']).count().sort_values(by= ['category_restaurant_id'],ascending = False).index[0:7].values


# loop through all price levels and categories to build prediction for each combination of those
for i in range(0, len(price_levels)):
    for j in range(0, len(frequent_categories)):
                
        
        # prepare feature values for success prediction
        df_location_values = pd.DataFrame({                        
                # Water distance
                'distance_to_water' : centroid_social_values['distance_to_water'],
                
                #Restaurant density
                'restaurant_density': centroid_social_values['restaurant_density'],
                
                #Crimes
                'crime_count': centroid_social_values['Criminal_cases'],
                
                #social values
                'population': centroid_social_values['population'],
                'share_under_18_years_old': centroid_social_values['share_under_18_years_old'],
                'share_65_year_olds_and_older': centroid_social_values['share_65_year_olds_and_older'],
                'share_foreigners': centroid_social_values['share_foreigners'],
                'population_with_migration_background': centroid_social_values['population_with_migration_background'],
                'share_population_with_migration_background': centroid_social_values['share_population_with_migration_background'],
                'households' : centroid_social_values['households'],
                'share_one_person_households' : centroid_social_values['share_one_person_households'],
                'share_households_with_children': centroid_social_values['share_households_with_children'],
                'share_households_with_single_parents' : centroid_social_values['share_households_with_single_parents'],
                'population_density' : centroid_social_values['population_density'],
                'employment_quote_in_%': centroid_social_values['employment_quote_in_%'],
                'share_unemployed_people': centroid_social_values['share_unemployed_people'],
                'sum_of_incomes_per_tax_reliable_person_in_EUR': centroid_social_values['sum_of_incomes_per_tax_reliable_person_in_EUR'],
                'prices_for_properties' : centroid_social_values['prices_for_properties'],
                'prices_for_condominiums' : centroid_social_values['prices_for_condominiums'],
                'share_students_in_Gymnasium' : centroid_social_values['share_students_in_Gymnasium'],
                'car_density' : centroid_social_values['car_density'],
            
          })
        df_categories_new = []
        prices = []
        
        for k in range(0,len(centroid_social_values.iloc[:,0])):
        
           df_categories_new.append(labelencoder_category.transform([frequent_categories[j]]))  
           prices.append(labelencoder_price.transform([price_levels[i]]))
           
        df_categories_new = np.array(df_categories_new)
        df_categories_binarized = onehotencoder.transform(df_categories_new.reshape(-1,1)).toarray()
        
        
        df_new_features = pd.concat(objs = [pd.DataFrame(prices),
                                            pd.DataFrame(df_location_values),
                                            pd.DataFrame(df_categories_binarized)],
                                                axis=1, ignore_index = True)
        
        # correct wrongly entered values from data source
        for l in range(len(df_new_features.columns)):
            feature_column = df_new_features.iloc[:,l]
            feature_column = pd.to_numeric(feature_column, errors='coerce')
            df_new_features[df_new_features.columns[l]] = feature_column
        
        # Fill missing data
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        df_new_features = imputer.fit_transform(df_new_features)
        
        # scale numerical values
        sc_X = StandardScaler()
        df_features = sc_X.fit_transform(df_new_features)
        
        # predict values for the combination of a certain price level and category with all locations
        rf_y_pred = rf_regressor.predict(df_new_features)
        
        
        # add prediction values to a comprising dataframe
        if ((i==0) and (j==0)):
            rf_y_pred_locations = pd.DataFrame({
                        'x': centroid_social_values['x'],
                        'y': centroid_social_values['y'],
                        'price_level': labelencoder_price.inverse_transform(np.reshape(prices,-1)),
                        'category': labelencoder_category.inverse_transform(np.reshape(df_categories_new,-1)),
                        'predicted_success': rf_y_pred
                    })
        else:
            rf_y_pred_locations = pd.concat([rf_y_pred_locations,
                                             pd.DataFrame({
                        'x': centroid_social_values['x'],
                        'y': centroid_social_values['y'],
                        'price_level': labelencoder_price.inverse_transform(np.reshape(prices,-1)),
                        'category': labelencoder_category.inverse_transform(np.reshape(df_categories_new,-1)),
                        'predicted_success': rf_y_pred
                    })],axis=0, ignore_index = True)
        
### Find the maximum success predictions for each location
df_category_maps = []

for i in range(0, len(frequent_categories)):
    df_recommendation = pd.DataFrame(columns= ['x','y', 'price_level','category','predicted_success'])
    
    filtered_category = rf_y_pred_locations[rf_y_pred_locations['category'] == frequent_categories[i]]
    
    for j in range(0,len(centroid_social_values)):
        x = centroid_social_values.iloc[j,0]
        y = centroid_social_values.iloc[j,1]
        filtered_x = filtered_category[filtered_category['x'] == x]
        filtered_y = filtered_x[filtered_x['y'] == y]
        max_pred = -99999.0
        candidate_found = False
        for i in range(0,len(filtered_y)):       
            if (filtered_y.iloc[i,4] > max_pred):
                max_pred = filtered_y.iloc[i,4]
                candidate_found = True
                pred_candidate = filtered_y.iloc[i,:].to_frame().transpose()
        if (candidate_found == True):
            df_recommendation = pd.concat([df_recommendation, pred_candidate],
                                          axis = 0,
                                          ignore_index = True)
            print('recommendation added')
            
        else:
            print('no recommendation found')
        
    df_category_maps.append(df_recommendation)
                
for i in range(0, len(df_category_maps)):
    social_values = pd.DataFrame({    
                # x and y
                'x': centroid_social_values['x'],
                'y': centroid_social_values['y'],
                
                # Water distance
                'distance_to_water' : centroid_social_values['distance_to_water'],
                
                #Restaurant density
                'restaurant_density': centroid_social_values['restaurant_density'],
                
                #Crimes
                'crime_count': centroid_social_values['Criminal_cases'],
                
                #social values
                'population': centroid_social_values['population'],
                'share_under_18_years_old': centroid_social_values['share_under_18_years_old'],
                'share_65_year_olds_and_older': centroid_social_values['share_65_year_olds_and_older'],
                'share_foreigners': centroid_social_values['share_foreigners'],
                'population_with_migration_background': centroid_social_values['population_with_migration_background'],
                'share_population_with_migration_background': centroid_social_values['share_population_with_migration_background'],
                'households' : centroid_social_values['households'],
                'share_one_person_households' : centroid_social_values['share_one_person_households'],
                'share_households_with_children': centroid_social_values['share_households_with_children'],
                'share_households_with_single_parents' : centroid_social_values['share_households_with_single_parents'],
                'population_density' : centroid_social_values['population_density'],
                'employment_quote_in_%': centroid_social_values['employment_quote_in_%'],
                'share_unemployed_people': centroid_social_values['share_unemployed_people'],
                'sum_of_incomes_per_tax_reliable_person_in_EUR': centroid_social_values['sum_of_incomes_per_tax_reliable_person_in_EUR'],
                'prices_for_properties' : centroid_social_values['prices_for_properties'],
                'prices_for_condominiums' : centroid_social_values['prices_for_condominiums'],
                'share_students_in_Gymnasium' : centroid_social_values['share_students_in_Gymnasium'],
                'car_density' : centroid_social_values['car_density'],
            
          })
    
    df_category_maps[i]['x'] = pd.to_numeric(df_category_maps[i]['x'])
    df_category_maps[i]['y'] = pd.to_numeric(df_category_maps[i]['y'])
   
    social_values['x'] = pd.to_numeric(social_values['x'])
    social_values['y'] = pd.to_numeric(social_values['y'])
    
    category_map = pd.merge(left = df_category_maps[i], right = social_values,
                            left_on = ['x','y'], right_on = ['x','y'])
    
    category_map = category_map.sort_values(by = ['predicted_success'], ascending = False)
    
    category_map.to_csv(github_path + '/data/recommendation_grid/recommendations_per_category/' + frequent_categories[i] + '_recommendation_centroids.csv')

    category_map.iloc[0:5,:].to_csv(github_path + '/data/recommendation_grid/recommendations_per_category/' + frequent_categories[i] + '_top_5_centroids.csv')


#################### Exploratory Data Analysis ########################


sns.boxplot(x= "price", y= "success", data = df_all.fillna('unknown'), order = ['unknown','€','€€','€€€','€€€€','€€€€€'])

#sns.boxplot(x= labelencoder_category.inverse_transform('hotdogs'), y= "success", data = df_all)

sns.lineplot(x= "prices_for_condominiums", y= "success", data = df_all)


sns.lineplot(x= "share_one_person_households", y= "success", data = df_all)

sns.lineplot(x= "population_density", y= "success", data = df_all)

sns.lineplot(x= "share_foreigners", y= "success", data = df_all)

sns.lineplot(x= "sum_of_incomes_per_tax_reliable_person_in_EUR", y= "success", data = df_all)

#df_all['sum_of_incomes_per_tax_reliable_person_in_EUR'].count()

sns.lineplot(x= "sum_of_incomes_per_tax_reliable_person_in_EUR", y= "success", data = df_all)


sns.lineplot(x= "density", y= "success", data = df_all)


sns.relplot(x= "distance_to_water", y= "success", kind = "line", data = df_all)

sns.relplot(x= "share_households_with_single_parents", y= "success", kind = "line", data = df_all)

# compare different ages
df_under_18 = pd.DataFrame({'age_group': ['Under 18 years old'] * len(df_all.index),
                            'share': df_all['share_under_18_years_old'],
                            'success': df_all['success']
                            })

df_above_65 = pd.DataFrame({'age_group': ['65 years old and older'] * len(df_all.index),
                            'share': df_all['share_65_year_olds_and_older'],
                            'success': df_all['success']
                            })
    
df_middle_age = pd.DataFrame({'age_group': ['18 to 65 years old'] * len(df_all.index),
                            'share': [100] * len(df_all.index) - df_all['share_65_year_olds_and_older'] - df_all['share_under_18_years_old'],
                            'success': df_all['success']
                            })
    
df_age_groups= pd.concat([df_under_18,df_above_65],axis = 0)
    
sns.lineplot(x= "share", y= "success", data = df_middle_age)



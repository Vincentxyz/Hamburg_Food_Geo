import pandas as pd
import numpy as np
from sklearn import preprocessing

df_restaurants = pd.read_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/Restaurants_in_Hamburg.csv')
df_social_values = pd.read_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/Social_Values_2016_adjusted.csv')


df_all = pd.merge(left=df_restaurants, right=df_social_values,
                left_on='Stadtteil', right_on = 'Stadtgebiet')

scaler = preprocessing.StandardScaler()

normalized_rating = scaler.fit_transform(np.array(df_all['rating']).reshape(-1,1))

normalized_review_count =scaler.fit_transform(np.array(df_all['review_cou']).reshape(-1,1))
    
df_all['success'] = normalized_rating + normalized_review_count

#businesses['success'] = succes

df_features = pd.DataFrame({
        'id': df_all['id'],
        'price' : df_all['price'].fillna('unknown'),
        
        #social values
        'population': df_all['Bevölkerung'],
        'share_under_18_years_old': df_all['Unter 18-Jährige'],
        '65_year_olds_and_older': df_all[' 65-Jährige und Ältere'],
        'share_65_year_olds_and_older': df_all['Anteil der 65-Jährigen und Älteren in %'],
        'foreigners': df_all['Ausländerinnen und Ausländer'],
        'share_foreigners': df_all['Ausländeranteil in %'],
        'population_with_migration_background': df_all['Bevölkerung mit Migrations-hintergrund'],
        'share_population_with_migration_background': df_all['Anteil der Bevölkerung mit Migrations-hintergrund in %'],
        'below_18_year_olds_with_migration_background' : df_all['Unter 18-Jährige mit Migrations-hintergrund'],
        'share_below_18_year_olds_with_migration_background': df_all['Anteil der unter 18-Jährigen mit Migrations-hintergrund in %'],
        'households' : df_all['Haushalte'],
        'people_per_household' : df_all['Personen je Haushalt'],
        'one_person_households' : df_all['Einpersonen-haushalte'],
        'share_one_person_households' : df_all['Anteil der Einpersonen-haushalte in %'],
        'households_with_children' : df_all['Haushalte mit Kindern'],
        'share_households_with_children': df_all['Anteil der Haushalte mit Kindern in %'],
        'single_parents' : df_all['Alleinerziehende'],
        'share_households_with_single_parents' : df_all['Anteil der Haushalte von Alleinerziehenden in %'],
        'area_in_square_kms' : df_all['Fläche in km²'],
        'population_density' : df_all['Bevölkerungs-dichte'],
        'births': df_all['Geburten'],
        'deaths': df_all['Sterbefälle'],
        'immigration': df_all['Zuzüge'],
        'emigration': df_all['Fortzüge'],
        'migration_balance': df_all['Wanderungssaldo'],
        'insurable_employees': df_all['Sozial-versicherungs-pflichtig Beschäftigte (Dez 2016)'],
        'employment_quote_in_%': df_all['Beschäftigten-quote in % (Dez 2016)'],
        'unemployed_people': df_all['Arbeitslose (Dez 2016)'],
        'share_unemployed_people': df_all['Arbeitslosenanteil in % (Dez 2016)'],
        'younger_unemployed_people' : df_all['Jüngere Arbeitslose (Dez 2016)'],
        'share_younger_unemployed_people': df_all['Arbeitslosenanteil Jüngerer in % (Dez 2016)'],
        'older_unemployed_people': df_all['Ältere Arbeitslose (Dez 2016)'],
        'share_older_unemployed_people': df_all['Arbeitslosenanteil Älterer in % (Dez 2016)'],
        'service_recipients_per_SGB_II' : df_all['Leistungs-empfänger/-innen nach SGB II (Dez 2016)'],
        'share_service_recipients_per_SGB_II': df_all['Anteil der Leistungs-empfänger/-innen nach SGB II in % (Dez 2016)'],
        'under_15_year_olds_in_minimum_income': df_all['Unter 15-Jährige in Mindestsicherung (Dez 2016)'],
        'share_under_15_year_olds_in_minimum_income': df_all['Anteil der unter 15-Jährigen in Mindestsicherung in % (Dez 2016)'],
        'shared_households_per_SGB_II': df_all['Bedarfs-gemeinschaften nach SGB II (Dez 2016)'],
        'tax_liable_people' : df_all['Lohn- und Einkommen-steuerpflichtige (2013)'],
        'sum_of_incomes_per_tax_reliable_person_in_EUR': df_all['Gesamtbetrag der Einkünfte je Steuerpflichtigen in EUR (2013)'],
        'residential_buildings' : df_all['Wohngebäude (2016)'],
        'apartments' : df_all['Wohnungen (2016)'],
        'apartments_ready_for_occupancy' : df_all['Bezugsfertige Wohnungen (2016)'],
        'homes_in_one_or_two_family_houses': df_all['Wohnungen in Ein- und Zweifamilien-häusern (2016)'],
        'share_homes_in_one_or_two_family_houses': df_all['Anteil der Wohnungen in Ein- und Zweifamilien-häusern in % (2016)'],
        'home_size_in_sqm': df_all['Wohnungsgröße in m² (2016)'], 
        'living_area_per_resident_in_sqm' : df_all['Wohnfläche je Einwohner/-in in m² (2016)'],
        'social_homes': df_all['Sozialwohnungen (Jan 2017)'],
        'share_social_homes' : df_all['Sozialwohnungs-anteil in % (Jan 2017)'],
        'social_homes_with_ends_of_commitment_until_2022' : df_all['Sozialwohnungen mit Bindungsauslauf bis 2022'],
        'share_social_homes_with_ends_of_commitment_until_2022': df_all['Sozialwohnungen mit Bindungsauslauf bis 2022 in %'],
        'prices_for_properties' : df_all['Preise für Grundstücke in EUR/m² (Jan 2017)'],
        'prices_for_one_or_two_family_houses_in_EUR/sqm' : df_all['Preise für Ein- bzw Zwei-familienhäuser in EUR/m² (Jan 2017)'],
        'prices_for_condominiums' : df_all['Preise für Eigentums-wohnungen in EUR/m² (Jan 2017)'],
        'kindergardens_and_preschool_classes': df_all['Kindergärten und Vorschulklassen (März 2017)'],
        'primary_schools': df_all['Grundschulen (2016/2017)'],
        'students_in_secondary_level_1': df_all['Schülerinnen und Schüler der Sekundarstufe I (2016/2017)'],
        'share_students_in_city_district_schools' : df_all['Anteil der Schülerinnen und Schüler in Stadtteilschulen in % (2016/2017)'],
        'share_students_in_Gymnasium' : df_all['Anteil der Schülerinnen und Schüler in Gymnasien in % (2016/2017)'],
        'residential_practitioners': df_all['Niedergelassene Ärzte (Jan 2017)'],
        'general_practitioners': df_all['Allgemeinärzte (Jan 2017)'],
        'dentists': df_all['Zahnärzte (Dez 2016)'],
        'pharmacys': df_all['Apotheken (Dez 2016)'],
        'private_cars': df_all['Private PKW (Jan 2017)'],
        'car_density' : df_all['PKW-Dichte (Jan 2017)']
  })

df_features.set_index('id', inplace=True,drop=True)


# Add categories

df_categories = pd.read_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/extract_categories.csv')


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

#Fill missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
df_features = imputer.fit_transform(df_features)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_features, df_all['success'], test_size = 0.2,random_state = 0)

# scale numerical values
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Random Forest Regression to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

github_path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/'

path = github_path + '/data/restaurants/'

businesses = pd.read_csv(path + 'Restaurants_in_Hamburg.csv')

## Standard Scaling ## 

scaler = preprocessing.StandardScaler()

standardized_rating = scaler.fit_transform(np.array(businesses['rating']).reshape(-1,1))


standardized_review_count =scaler.fit_transform(np.array(businesses['review_cou']).reshape(-1,1))
    
success = standardized_rating + standardized_review_count

businesses['success'] = success



### Normalizing ## 
#
#scaler = preprocessing.Normalizer()
#
#normalized_rating = scaler.fit_transform(np.array(businesses['rating']).reshape(-1,1))
#
#normalized_review_count =scaler.fit_transform(np.array(businesses['review_cou']).reshape(-1,1))
#    
#success = normalized_rating * 5.0 + normalized_review_count * 5.0
#
#businesses['success'] = success
#

businesses.to_csv(path + 'Restaurants_in_Hamburg.csv')


plt.hist(normalized_rating)

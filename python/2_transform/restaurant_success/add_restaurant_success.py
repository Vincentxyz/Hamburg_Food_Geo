import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

path = 'C:/Users/vince_000/Documents/GitHub/Hamburg_Food_Geo/data/restaurants/'

businesses = pd.read_csv(path + 'Restaurants_in_Hamburg.csv')


scaler = preprocessing.StandardScaler()

standardized_rating = scaler.fit_transform(np.array(businesses['rating']).reshape(-1,1))


standardized_review_count =scaler.fit_transform(np.array(businesses['review_cou']).reshape(-1,1))
    
success = standardized_rating + standardized_review_count

businesses['success'] = success

plt.hist(success)

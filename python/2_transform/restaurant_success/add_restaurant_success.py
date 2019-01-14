import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

businesses = pd.read_csv('C:/Users/vince_000/Documents/Geotesting/Test_Files/Hamburg/CSV/Restaurants_in_Hamburg.csv')


scaler = preprocessing.StandardScaler()

normalized_rating = scaler.fit_transform(np.array(businesses['rating']).reshape(-1,1))

normalized_review_count =scaler.fit_transform(np.array(businesses['review_count']).reshape(-1,1))
    
success = normalized_rating + normalized_review_count

businesses['success'] = success

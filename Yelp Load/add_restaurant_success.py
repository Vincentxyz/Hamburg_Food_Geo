import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

businesses = pd.read_csv('extract_businesses.csv')

#plt.hist(businesses.iloc[:,3])


#Grouping accroding to pricing
businesses['price'].unique()

is_price_level = []

is_price_level.append(businesses['price'].isna())
is_price_level.append(businesses['price'] == '€')
is_price_level.append(businesses['price'] == '€€')
is_price_level.append(businesses['price'] == '€€€')
is_price_level.append(businesses['price'] == '€€€€')

businesses['success'] = 0
normalized_rating = []
normalized_review_count = []
restaurant_sucess = []

for i in range(len(is_price_level)):
    rankings = []
    review_counts = []
    for j in range(len(is_price_level[i])):
        if is_price_level[i][j] == True:
            rankings.append(businesses.iloc[j,3])
            review_counts.append(businesses.iloc[j,4])
    #df_ranking_array = businesses.iloc[:,3]
    ranking_array = np.array(rankings)
    ranking_array = ranking_array.reshape(-1,1)
    scaler = preprocessing.StandardScaler()
    normalized_rating = scaler.fit_transform(ranking_array)
        
    review_count_array = np.array(review_counts)
    review_count_array = review_count_array.reshape(-1,1)
    scaler = preprocessing.StandardScaler()
    normalized_review_count =scaler.fit_transform(review_count_array)
    
    success = normalized_rating + normalized_review_count
    
    k = 0
    
    for j in range(len(is_price_level[i])):
        if is_price_level[i][j] == True:
            businesses.iloc[j, 10] = success[k][0]
            k = k+1
        
businesses.to_csv('extract_businesses.csv')
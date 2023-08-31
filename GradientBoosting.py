# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:10:44 2023

@author: SergeyHSE
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

! wget https://www.dropbox.com/s/psutl0zafq50828/data.tsv > ./data.tsv
data = pd.read_csv('./data.tsv', sep='\t')
pd.set_option('display.max_columns', None)
data.head()

# Let's break our dataset into categorical and numeric features,
# and for now we will work only with numeric ones

num_features = ['agent_fee', 'floor', 'floors_total',
                'kitchen_area', 'living_area', 'price',
                'rooms_offered', 'total_area', 'total_images']
cat_features = ['balcony', 'building_type', 'month', 'renovation', 'studio']

X_train, X_test, y_train, y_test = train_test_split(data[num_features], data['exposition_time'],
                                                    test_size=0.3, shuffle=False)

#1
# Let's train the implementation of gradient boosting Lightgdm and Cart boost on numerical features
# without parameter selection
# Then we calculate MAE and compare its between LightGBM and Catboost

lg = LGBMRegressor(random_seed=0)
%time lg.fit(X_train, y_train)
y_pred_lg = lg.predict(X_test)
mae_lg = mean_absolute_error(y_test, y_pred_lg)
print(mae_lg)

cbr = CatBoostRegressor(random_seed=0, loss_function='MAE')
%time cbr.fit(X_train, y_train)
y_pred_cbr = cbr.predict(X_test)
mae_cbr = mean_absolute_error(y_test, y_pred_cbr)

answer1 = (mae_lg - mae_cbr)
print(answer1)

#2
# We are trying to select the optimal parameters for Catboost  using all available combinations of:
# tree depths {5, 7, 9};
# learning rate {0.05, 0.1, 0.5}
# We will do it both methods: GridSearchCV and embedded in Catboost

from sklearn.model_selection import GridSearchCV

grid_searcher = GridSearchCV(CatBoostRegressor(random_seed=0, loss_function='MAE'),
                             param_grid = {'depth' : [5, 7, 9],
                                           'learning_rate' : [0.05, 0.1, 0.5]},
                             scoring='neg_mean_absolute_error')
grid_searcher.fit(X_train, y_train)
grid_searcher.best_params_

cbr = CatBoostRegressor(depth=5, learning_rate=0.05,
                        random_seed=0, loss_function='MAE')
%time cbr.fit(X_train, y_train)
y_pred_cbr = cbr.predict(X_test)
mean_absolute_error(y_test, y_pred_cbr)

model = CatBoostRegressor(random_seed=0, loss_function='MAE')
model.fit(X_train, y_train, verbose=False, plot=True)
grid = {'depth' : [5, 7, 9],
        'learning_rate' : [0.05, 0.1, 0.5]}

grid_search_result = model.grid_search(grid, X_train, y_train, plot=True)
print(grid_search_result)
cbr = CatBoostRegressor(depth=7, learning_rate=0.05, random_seed=0, loss_function='MAE')
%time cbr.fit(X_train, y_train)
y_pred_cbr = cbr.predict(X_test)
y_pred_train = cbr.predict(X_train)
mean_absolute_error(y_test, y_pred_cbr)
feature_imp = cbr.feature_importances_

#3
# Now we calculate MAE by Catboost for numeric and categorial features


X_train, X_test, y_train, y_test = train_test_split(data[num_features+cat_features],
                                                    data['exposition_time'], test_size=0.3, shuffle=False)

model = CatBoostRegressor(random_seed=0, loss_function='MAE', cat_features=cat_features)
grid = {'depth' : [5, 7, 9],
        'learning_rate' : [0.05, 0.1, 0.5]}

grid_search_result = model.grid_search(grid, X_train, y_train, plot=True)
print(grid_search_result)
cbr_all = CatBoostRegressor(depth=9, learning_rate=0.05, random_seed=0,
                        loss_function='MAE', cat_features=cat_features)

%time cbr_all.fit(X_train, y_train)
y_pred_cbr_all = cbr_all.predict(X_test)
y_train_pred_all = cbr_all.predict(X_train)
mae_cbr_all = mean_absolute_error(y_test, y_pred_cbr_all)
print(mae_cbr_all)
feature_imp_all = cbr_all.feature_importances_

feature_imp
feature_imp_all

y_pred_train.shape
y_pred_cbr.shape
feature_imp.shape
y_train_pred_all.shape
y_pred_cbr_all.shape
feature_imp_all.shape


#4
# we wil implement blending (getting responses from several models and taking them with weights
# (they need to be selected on a training sample)) the models obtained in tasks 2 and 3
# and output MAE on the test sample.

def select_weights(y_true, y_pred_1, y_pred_2):
    metric = []
    np.random.seed(0)
    grid = np.linspace(-1, 1, 1000)
    for w_0 in grid:
        if w_0 < 0:
            w_1 = -(1 + w_0)
        elif w_0 >= 0:
            w_1 = 1 - w_0
        y = y_pred_1*w_0 + y_pred_2*w_1
        metric.append([mean_absolute_error(y_true, y), w_0, w_1])
    return metric

mae_cbr_blending_train, w_0, w_1 = min(select_weights(y_train, y_pred_train, y_train_pred_all),
                                    key=lambda x: x[0])

mae_cbr_blending_train
w_0
w_1

print('test mae blending = %.4f' % mean_absolute_error(y_test, y_pred_cbr*w_0 + y_pred_cbr_all*w_1))

from numpy.random import rand, randn

def select_weights_feat(y_true, y_pred_1, y_pred_2):
    metric =[]
    np.random.seed(0)
    grid = np.random.rand(1000)
    grid_all = np.random.rand(1000)
    for w_0 in grid:
        w_0 = w_0
    for w_1 in grid_all:
        w_1 = w_1
        
    y = y_pred_1*w_0 + y_pred_2*w_1
    metric.append([mean_absolute_error(y_true, y), w_0, w_1])
    return metric

mae_cbr_blending, w_0, w_1 = min(select_weights_feat(y_train, y_pred_train, y_train_pred_all),
                                    key=lambda x: x[0])

mae_cbr_blending
w_0
w_1
print('test mae blending = %.4f' % mean_absolute_error(y_test, y_pred_cbr*w_0 + y_pred_cbr_all*w_1))

#5
# In Task 3, we selected hyperparameters for CatBoost on all signs.
# We are visualizing their importance in the form of a horizontal barplot,
# Then we will sort the signs in descending order of importance, sign the names of the signs along the y axis.

feature_imp_all


sorted_weights = sorted(zip(feature_imp_all.ravel(),
                            data[num_features+cat_features].columns), reverse=True)

sorted_weights
weights = [x[0] for x in sorted_weights]
weights
features = [x[1] for x in sorted_weights]
features

df = pd.DataFrame({'features':features, 'weights':weights})
df
ax = df.plot.barh(x='features', y='weights', rot=0)

# For each of the two algorithms, we will remove the unimportant features 
# and train the model with the same parameters on the remaining features.
# Output the difference between the MAE values on the test sample before and after removing the features.

num_features = ['agent_fee', 'floor', 'floors_total',
                'kitchen_area', 'living_area', 'price',
                'rooms_offered', 'total_area', 'total_images']
cat_features1 = ['balcony', 'building_type', 'month', 'renovation']
X_train, X_test, y_train, y_test = train_test_split(data[num_features+cat_features1], data['exposition_time'], test_size=0.3, shuffle=False)

cbr_cut = CatBoostRegressor(depth=9, learning_rate=0.05, random_seed=0,
                        loss_function='MAE', cat_features=cat_features1)

cbr_cut.fit(X_train, y_train)
y_pred_cut = cbr_cut.predict(X_test)
mae_cbr_cut = mean_absolute_error(y_test, y_pred_cut)
print(mae_cbr_cut)
answer5 = mae_cbr_all - mae_cbr_cut
print(answer5)

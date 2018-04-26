import numpy as np
import pandas as pd
import gc
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from dtypes import dtypes

use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os']

ITERATION = 1000
NUM_TRAIN = 20000000
NUM_CV = 10000000

X_day8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes, skiprows=range(1, 62945076-NUM_TRAIN))
X_day9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes, skiprows=range(1, 53016938-NUM_CV))

Y_day8 = X_day8["is_attributed"]
X_day8 = X_day8.drop(["is_attributed"], axis=1)
Y_day9 = X_day9["is_attributed"]
X_day9 = X_day9.drop(["is_attributed"], axis=1)

fout = open("best_feature.dat",'w')
fout.write(" Score, n_drop, droped\n")
array = [i for i in range(4, 29)]
best_score = 0.0

it = 0
while it < ITERATION:

    it += 1
    n_drop = np.random.randint(1, 25)
    to_drop = np.random.choice(array, n_drop, replace=False)
    X_train = X_day8.drop([use_features[i] for i in to_drop], axis=1)
    X_cv = X_day9.drop([use_features[i] for i in to_drop], axis=1)

    gbm_train = lgbm.Dataset(X_train, Y_day8, categorical_feature=['app','device','os','channel'])
    gbm_cv = lgbm.Dataset(X_cv, Y_day9, categorical_feature=['app','device','os','channel'])
    del X_train
    gc.collect()

    params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'goss',
        
        # Deal with overfitting
#        'bagging_fraction': 0.9, 
#        'bagging_freq': 1,
        'min_data_in_leaf': 5000,
        'feature_fraction': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'max_bin': 255,
        
        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 200,
    }

    bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    cv_pred = bst.predict(X_cv, num_iteration=bst.best_iteration)
    score = roc_auc_score(Y_day9, cv_pred)
    if score > best_score:
        best_score = score
    print(best_score)
    to_write = str(score) + ',' + str(n_drop) + ', ['
    for drop in to_drop:
        to_write = to_write + str(drop) + ','
    to_write += ']\n'
    fout.write(to_write)
    fout.flush()

    del X_cv
    gc.collect()



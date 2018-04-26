import numpy as np
import pandas as pd
import gc
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score

dtypes = {
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'is_attributed':'uint8',
    'total_click':'uint32',
    'click_per_hour':'uint32',
    'click_per_channel':'uint32',
    'click_per_app':'uint32',
    'click_per_device':'uint32',
    'click_per_os':'uint32',   
    'click_app_os':'uint32',
    'click_app_channel':'uint32',
    'click_os_channel':'uint32',
    'click_ip_app_os_channel_hour':'uint32',
    'next_click_ip':'float64',
    'next_click_ip_channel':'float64',
    'next_click_ip_app':'float64',
    'next_click_ip_device':'float64',
    'next_click_ip_os':'float64',
    'p_click_ip':'uint32',
    'p_click_ip_channel':'uint32',
    'p_click_ip_app':'uint32',
    'p_click_ip_device':'uint32',
    'p_click_ip_os':'uint32',
    'click_ip_app_os_device_hour':'uint32',
    'click_ip_app_os_device_minute':'uint32',
    'click_app_device':'uint32',
    'click_os_device':'uint32',
    'hour':'uint8',
    'click_device_channel':'uint32',
    'click_ip_app_hour':'uint32',
    'click_ip_os_hour':'uint32',
    'click_ip_device_hour':'uint32',
    'click_ip_channel_hour':'uint32',
    'click_ip_app_os':'uint32',
    'click_app':'uint32',
    'click_os':'uint32',
    'click_device':'uint32',
    'click_channel':'uint32',
    'next_click_ip_app_os_device':'float64',
    'next_click_ip_app_os':'float64',
    'next_click_app_channel':'float64',
    'p_click_ip_app_os_device':'uint32',
    'p_click_ip_app_os':'uint32',
    'tar_app':'float32',
    'tar_os':'float32',
    'tar_device':'float32',
    'tar_channel':'float32'
}

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

gbm_train = lgbm.Dataset(X_day8, Y_day8, categorical_feature=['app','device','os','channel','hour'])
gbm_cv = lgbm.Dataset(X_day9, Y_day9, categorical_feature=['app','device','os','channel','hour'])

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
#    bst.save_model('model.txt', num_iteration=bst.best_iteration)
cv_pred = bst.predict(X_day9, num_iteration=bst.best_iteration)
score = roc_auc_score(Y_day9, cv_pred)
print(score)



import pandas as pd
import numpy as np
import gc

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
    'f_click_ip':'uint32',
    'p_click_ip_channel':'uint32',
    'f_click_ip_channel':'uint32',
    'p_click_ip_app':'uint32',
    'f_click_ip_app':'uint32',
    'p_click_ip_device':'uint32',
    'f_click_ip_device':'uint32',
    'p_click_ip_os':'uint32',
    'f_click_ip_os':'uint32',
    'click_ip_app_os_device_hour':'uint32',
    'click_ip_app_os_device_minute':'uint32',
    'click_app_device':'uint32',
    'click_os_device':'uint32',
    'ip_minute_ave':'float32',
    'ip_minute_std':'float32',
    'ins_minute_ave':'float32',
    'ins_minute_std':'float32'
}

use_features = ['app','os','channel','total_click','click_app_os','click_app_channel','click_os_channel','click_ip_app_os_channel_hour',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute','click_app_device','click_os_device','ins_minute_ave']
use_base = ['app','os','channel','is_attributed']
use_base_test = ['app','os','channel']
use_group = ['total_click','click_app_os','click_app_channel','click_os_channel','click_ip_app_os_channel_hour']
use_add = ['click_ip_app_os_device_hour','click_ip_app_os_device_minute','click_app_device','click_os_device','ins_minute_ave']

# Prepare train
train_base = pd.read_csv("../feature/train-day8.csv", dtype=dtypes, usecols=use_base)
Y_train = train_base['is_attributed'].values
train_base = train_base.drop(['is_attributed'], axis=1)
train_group = pd.read_csv("../feature/train-day8-groupfeature.csv", dtype=dtypes, usecols=use_group)
train_add = pd.read_csv("../feature/train-day8-groupadd.csv", dtype=dtypes, usecols=use_add)
#train_next = pd.read_csv("../feature/train-day8-nextclick.csv", dtype=dtypes)
#train_pf = pd.read_csv("../feature/train-day8-pfclick.csv",dtype=dtypes)
#X_train = np.concatenate([train_base.values, train_group.values, train_next.values, train_pf], axis=1)
#del train_base, train_group, train_next, train_pf
X_train = np.concatenate([train_base.values, train_group.values, train_add.values], axis=1)
del train_base, train_group, train_add
gc.collect()
print("Finished loading train!")

# Prepare validation
cv_base = pd.read_csv("../feature/train-day9.csv", dtype=dtypes, usecols=use_base)
Y_cv = cv_base['is_attributed'].values
cv_base = cv_base.drop(['is_attributed'], axis=1)
cv_group = pd.read_csv("../feature/train-day9-groupfeature.csv", dtype=dtypes, usecols=use_group)
cv_add = pd.read_csv("../feature/train-day9-groupadd.csv", dtype=dtypes, usecols=use_add)
#cv_next = pd.read_csv("../feature/train-day9-nextclick.csv", dtype=dtypes)
#cv_pf = pd.read_csv("../feature/train-day9-pfclick.csv",dtype=dtypes)
#X_cv = np.concatenate([cv_base.values, cv_group.values, cv_next.values, cv_pf.values], axis=1)
#del cv_base, cv_group, cv_next, cv_pf
X_cv = np.concatenate([cv_base.values, cv_group.values, cv_add.values], axis=1)
del cv_base, cv_group
gc.collect()
print("Finished loading train!")


import lightgbm as lgbm

gbm_train = lgbm.Dataset(X_train, Y_train, feature_name=use_features,\
                 categorical_feature=['app','os','channel'])
gbm_cv = lgbm.Dataset(X_cv, Y_cv, feature_name=use_features,\
                categorical_feature=['app','os','channel'])
    
del X_train, Y_train, X_cv, Y_cv
gc.collect()

params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'gbdt',
        
        # Deal with overfitting
        'bagging_fraction': 0.9, 
        'bagging_freq': 1,
        'min_data_in_leaf': 5000,
        'feature_fraction': 0.8,
        'num_leaves': 31,
        'max_depth': -1,
        'max_bin': 255,
        
        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 200
}

bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)


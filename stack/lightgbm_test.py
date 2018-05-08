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
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os',
               'ip_uniq_app','ip_os_device_uniq_app','ip_uniq_device','ip_uniq_channel','ip_app_uniq_os','app_uniq_channel']

#X_day8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes)
#X_day9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes)
#Y_day8 = X_day8["is_attributed"]
#X_day8 = X_day8.drop(["is_attributed"], axis=1)
#Y_day9 = X_day9["is_attributed"]
#X_day9 = X_day9.drop(["is_attributed"], axis=1)

gbm_train = lgbm.Dataset("../feature/train-8.bin", feature_name=use_features, categorical_feature=['app','device','os','channel','hour'])
gbm_cv = lgbm.Dataset("../feature/train-9.bin", feature_name=use_features, categorical_feature=['app','device','os','channel','hour'])

params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.029852707308510206,
        'num_iterations': 1000,
        'boosting' : 'goss',
        
        # Deal with overfitting
#        'bagging_fraction': 0.9, 
#        'bagging_freq': 1,
        'min_data_in_leaf': 5957,
        'feature_fraction': 0.8101545949032861,
        'num_leaves': 50,
        'max_depth': 7,
        'max_bin': 1000,
        
        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 237.38377831205426,
}

bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)

del gbm_train, gbm_cv
gc.collect()

X_day8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes)
Y_day8 = X_day8["is_attributed"]
X_day8 = X_day8.drop(["is_attributed"], axis=1)

train_pred = bst.predict(X_day8, num_iteration=bst.best_iteration)
del X_day8, Y_day8
gc.collect()

X_day9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes)
Y_day9 = X_day9["is_attributed"]
X_day9 = X_day9.drop(["is_attributed"], axis=1)
cv_pred = bst.predict(X_day9, num_iteration=bst.best_iteration)
train_frame = pd.DataFrame({'lightgbm':train_pred, 'label':Y_day8.values})
train_frame.to_csv('day8-lightgbm.csv', index=False)
cv_frame = pd.DataFrame({'lightgbm':cv_pred, 'label':Y_day9.values})
cv_frame.to_csv('day9-lightgbm.csv', index=False)



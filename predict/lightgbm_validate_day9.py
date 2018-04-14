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
    'f_click_ip_os':'uint32'
}


# Prepare train
train_base = pd.read_csv("../feature/train-day8.csv", dtype=dtypes, usecols=['app','device','os','channel','is_attributed'])
Y_train = train_base['is_attributed'].values
train_base = train_base.drop(['is_attributed'], axis=1)
train_group = pd.read_csv("../feature/train-day8-groupfeature.csv", dtype=dtypes)
train_next = pd.read_csv("../feature/train-day8-nextclick.csv", dtype=dtypes)
train_pf = pd.read_csv("../feature/train-day8-pfclick.csv",dtype=dtypes)
X_train = np.concatenate([train_base.values, train_group.values, train_next.values, train_pf], axis=1)
del train_base, train_group, train_next, train_pf
gc.collect()
print("Finished loading train!")

# Prepare validation
cv_base = pd.read_csv("../feature/train-day9.csv", dtype=dtypes, usecols=['app','device','os','channel','is_attributed'])
Y_cv = cv_base['is_attributed'].values
cv_base = cv_base.drop(['is_attributed'], axis=1)
cv_group = pd.read_csv("../feature/train-day9-groupfeature.csv", dtype=dtypes)
cv_next = pd.read_csv("../feature/train-day9-nextclick.csv", dtype=dtypes)
cv_pf = pd.read_csv("../feature/train-day9-pfclick.csv",dtype=dtypes)
X_cv = np.concatenate([cv_base.values, cv_group.values, cv_next.values, cv_pf.values], axis=1)
del cv_base, cv_group, cv_next, cv_pf
gc.collect()
print("Finished loading train!")

# Prepare test
test_base = pd.read_csv("../feature/test.csv", dtype=dtypes, usecols=['app','device','os','channel'])
test_group = pd.read_csv("../feature/test-groupfeature.csv", dtype=dtypes)
test_next = pd.read_csv("../feature/test-nextclick.csv", dtype=dtypes)
test_pf = pd.read_csv("../feature/test-pfclick.csv", dtype=dtypes)
X_test = np.concatenate([test_base.values, test_group.values, test_next.values, test_pf.values], axis=1)
del test_base, test_group, test_next, test_pf
gc.collect()

click_id = np.loadtxt("../feature/test_click_id.csv").astype(np.uint32)

#del train, test
#gc.collect()
print("Finshed loading data!")

import lightgbm as lgbm

gbm_train = lgbm.Dataset(X_train, Y_train, feature_name=['app','device','os','channel','total_click','click_per_hour',\
                'click_per_channel','click_per_app','click_per_device','click_per_os',\
                'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
                'p_click_ip','f_click_ip','p_click_ip_channel','f_click_ip_channel','p_click_ip_app','f_click_ip_app',\
                'p_click_ip_device','f_click_ip_device','p_click_ip_os','f_click_ip_os'],\
                 categorical_feature=['app','device','os','channel'])
gbm_cv = lgbm.Dataset(X_cv, Y_cv, feature_name=['app','device','os','channel','total_click','click_per_hour',\
                'click_per_channel','click_per_app','click_per_device','click_per_os',\
                'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
                'p_click_ip','f_click_ip','p_click_ip_channel','f_click_ip_channel','p_click_ip_app','f_click_ip_app',\
                'p_click_ip_device','f_click_ip_device','p_click_ip_os','f_click_ip_os'],\
                categorical_feature=['app','device','os','channel'])
    
del X_train, Y_train, X_cv, Y_cv
gc.collect()

params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'gbdt',
        
        # Deal with overfitting
        'bagging_fraction': 0.8, 
        'bagging_freq': 1,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'num_leaves': 11,
        'max_depth': -1,
        'max_bin': 100,
        
        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 200
}

bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)
ypred = bst.predict(X_test, num_iteration=bst.best_iteration)

submit = pd.DataFrame({'click_id':click_id, 'is_attributed':ypred})
submit.to_csv('submission.csv', index=False)


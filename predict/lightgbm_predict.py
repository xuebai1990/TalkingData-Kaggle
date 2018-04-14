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
    'next_click_ip_os':'float64'
}


# Prepare train
train_base = pd.read_csv("../feature/train-day8.csv", dtype=dtypes, usecols=['app','device','os','channel','is_attributed'])
Y_train = train_base['is_attributed'].values
train_base = train_base.drop(['is_attributed'], axis=1)
train_group = pd.read_csv("../feature/train-day8-groupfeature.csv", dtype=dtypes)
train_next = pd.read_csv("../feature/train-day8-nextclick.csv", dtype=dtypes)
X_train = np.concatenate([train_base.values, train_group.values, train_next.values], axis=1)
del train_base, train_group, train_next
gc.collect()
print("Finished loading train!")

# Prepare test
test_base = pd.read_csv("../feature/test.csv", dtype=dtypes, usecols=['app','device','os','channel'])
test_group = pd.read_csv("../feature/test-groupfeature.csv", dtype=dtypes)
test_next = pd.read_csv("../feature/test-nextclick.csv", dtype=dtypes)
X_test = np.concatenate([test_base.values, test_group.values, test_next.values], axis=1)
del test_base, test_group, test_next
gc.collect()

click_id = np.loadtxt("../feature/test_click_id.csv").astype(np.uint32)

#del train, test
#gc.collect()
print("Finshed loading data!")

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
import lightgbm as lgbm

sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
result = np.zeros(X_test.shape[0])
train_result = np.zeros(X_train.shape[0])

for (train_index, cv_index) in sss.split(X_train, Y_train):

    XX_train, YY_train = X_train[train_index], Y_train[train_index]
    XX_cv, YY_cv = X_train[cv_index], Y_train[cv_index]
    gbm_train = lgbm.Dataset(XX_train, YY_train, feature_name=['app','device','os','channel','total_click','click_per_hour',\
                'click_per_channel','click_per_app','click_per_device','click_per_os',\
                'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os'],\
                 categorical_feature=['app','device','os','channel'])
    gbm_cv = lgbm.Dataset(XX_cv, YY_cv, feature_name=['app','device','os','channel','total_click','click_per_hour',\
                'click_per_channel','click_per_app','click_per_device','click_per_os',\
                'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os'],\
              categorical_feature=['app','device','os','channel'])
    
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
        'num_threads': 8,
        'scale_pos_weight': 200
    }

    bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
    bst.save_model('model.txt', num_iteration=bst.best_iteration)
    ypred = bst.predict(X_test, num_iteration=bst.best_iteration)
    result += ypred

    train_pred = bst.predict(XX_cv, num_iteration=bst.best_iteration)
    train_result[cv_index] = train_pred

result /= 5


submit = pd.DataFrame({'click_id':click_id, 'is_attributed':result})
submit.to_csv('submit.csv', index=False)
train_result = pd.DataFrame({"LightGBM":train_result})
train_result.to_csv("train_predict.csv", index=False)


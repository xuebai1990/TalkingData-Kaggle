import numpy as np
import pandas as pd
import lightgbm as lgbm
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
import gc
from sklearn.model_selection import PredefinedSplit
from dtypes import dtypes

ITERATION = 1000
NUM_TRAIN = 24000000
NUM_CV = 10000000

train = pd.read_csv("../feature/train-day8-total.csv", skiprows=range(1, 62945076-NUM_TRAIN), dtype=dtypes)
cv = pd.read_csv("../feature/train-day9-total.csv", skiprows=range(1, 53016938-NUM_CV), dtype=dtypes)

Y_train = train["is_attributed"].values
X_train = train.drop(["is_attributed"], axis=1).values
Y_cv = cv["is_attributed"].values
X_cv = cv.drop(["is_attributed"], axis=1).values

N = X_train.shape[0]
X_total = np.concatenate([X_train, X_cv], axis=0)
Y_total = np.concatenate([Y_train, Y_cv], axis=0)
T = X_total.shape[0]
train_index = np.array([i for i in range(N)])
cv_index = np.array([i for i in range(N,T)])
del train, cv, X_train, Y_train, X_cv, Y_cv
gc.collect()

use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ipp_app_os']



#gbm_X = lgbm.Dataset(X_train, feature_name=use_features,\
#                 categorical_feature=['app','device','os','channel','hour'])
#gbm_Y = lgbm.Dataset(Y_train)

#del X_train, Y_train
#gc.collect()


# Classifier
bayes_cv = BayesSearchCV(
           estimator = lgbm.LGBMClassifier(
                       n_jobs = 16,
                       boosting_type='goss',
                       n_estimators=1000,
                       objective='binary',
                       ),
           search_spaces = {
                        'max_bin':(100, 1000),
                        'num_leaves':(2, 50),
                        'max_depth':(2, 20),
                        'learning_rate':(0.01, 1.0, 'log-uniform'),
                        'scale_pos_weight':(80, 500, 'log-uniform'),
                        'min_child_samples':(10, 10000),
                        'colsample_bytree':(0.1, 1.0, 'uniform'),
                        },
           fit_params = {
                    'feature_name':use_features,
                    'categorical_feature':['app','device','os','channel','hour'],
                    'early_stopping_rounds':10,
                    'eval_metric':'auc',
                    'eval_set':(X_total[N:T], Y_total[N:T]),
                    },
           scoring = 'roc_auc',
           cv = [(train_index, cv_index)],
           n_jobs = 1,
           n_iter = ITERATION,
           verbose = 0,
           random_state = 0
           )

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv.best_score_, 4),
        bayes_cv.best_params_
    ))
    
    # Save all model results
    all_models.to_csv("LGBM_cv_results.csv")

result = bayes_cv.fit(X_total, Y_total, callback=status_print)


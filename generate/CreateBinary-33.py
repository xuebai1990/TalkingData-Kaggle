import pandas as pd
import numpy as np
import gc
import lightgbm as lgbm
from dtypes import dtypes


use_features = ['app','device','os','channel','is_attributed','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os',\
               'ip_uniq_app','ip_os_device_uniq_app','ip_uniq_device','ip_uniq_channel','ip_app_uniq_os','app_uniq_channel']

# Prepare train
train_6 = pd.read_csv("../feature/train-day6-total.csv", dtype=dtypes).values
train_7 = pd.read_csv("../feature/train-day7-total.csv", dtype=dtypes).values
train_8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes).values
train_9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes).values
X_total = np.concatenate([train_6, train_7, train_8, train_9], axis=0)
del train_6, train_7, train_8, train_9
gc.collect()

X_total = pd.DataFrame(X_total, columns=use_features)
N = X_total.shape[0]
N_cv = 25000000
#Y_total = X_total["is_attributed"]
X_total = X_total.drop(["is_attributed"], axis=1)
X_train = X_total[:N-N_cv]
X_train.to_csv("../feature/train-all.csv", index=False)
del X_train
gc.collect()
#Y_train = Y_total[:N-N_cv]
X_cv = X_total[N-N_cv:]
X_cv.to_csv("../feature/cv-all.csv", index=False)
#Y_cv = Y_total[N-N_cv:]
#del X_total, Y_total
#gc.collect()
#Y_train = X_train['is_attributed'].values
#X_train = X_train.drop(["is_attributed"], axis=1).values
#print("Finished loading train!")

#gbm_train = lgbm.Dataset(X_train, Y_train, feature_name=use_features, categorical_feature=['app','device','os','channel','hour'])
#gbm_train.save_binary("../feature/train-8.bin")
#print("Finished save day 8 binary")

#gbm_train = lgbm.Dataset(X_train, Y_train, feature_name=use_features,categorical_feature=['app','device','os','channel','hour'])
#gbm_train.save_binary("../feature/train-total.bin")
#print("Finished loading train")

# Prepare train
#X_train = pd.read_csv("../feature/train-day7-total.csv", dtype=dtypes)
#Y_train = X_train['is_attributed'].values
#X_train = X_train.drop(["is_attributed"], axis=1).values
#print("Finished loading train!")

#gbm_train = lgbm.Dataset(X_train, Y_train, feature_name=use_features,\
#                 categorical_feature=['app','device','os','channel','hour'])
#gbm_train.save_binary("../feature/train-7.bin")
#print("Finished save day 7 binary")
#del X_train, Y_train, gbm_train
#gc.collect()

# Prepare validation
#X_cv = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes)
#Y_cv = X_cv['is_attributed'].values
#X_cv = X_cv.drop(["is_attributed"], axis=1)
#print("Finished loading cv!")

#gbm_cv = gbm_train.create_valid(X_cv, Y_cv)
#gbm_cv.save_binary("../feature/train-9.bin")
#print("Finished save day 9 binary")
#                categorical_feature=['app','device','os','channel','hour'])
#gbm_cv.save_binary("../feature/train-9.bin")
#print("Finished save day 9 binary")   

# Prepare test
#X_7 = pd.read_csv("../feature/train-day7-total.csv", dtype=dtypes)
#Y_7 = X_7["is_attributed"]
#X_7 = X_7.drop(["is_attributed"], axis=1)
#print("Finished loading test!")

#gbm_cv = gbm_train.create_valid(X_cv, Y_cv)
#gbm_cv.save_binary("../feature/cv-total.bin")
#print("Finished save cv binary")



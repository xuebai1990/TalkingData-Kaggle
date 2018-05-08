import pandas as pd
import numpy as np
import gc
import lightgbm as lgbm

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
    'ins_minute_std':'float32',
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

#use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel','click_os_channel','click_ip_app_os_channel_hour',\
#               'click_ip_app_os_device_hour','click_ip_app_os_device_minute','click_app_device','click_os_device','ins_minute_ave',\
#               'hour','click_device_channel','click_ip_app_os']
all_features = ['app','device','os','channel','is_attributed','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os']

individuals = [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]
use_features = [all_features[i] for i in range(len(individuals)) if individuals[i] == 1]


# Prepare train
train_6 = pd.read_csv("../feature/train-day6-total.csv", dtype=dtypes, usecols=use_features).values
train_7 = pd.read_csv("../feature/train-day7-total.csv", dtype=dtypes, usecols=use_features).values
train_8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes, usecols=use_features).values
train_9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes, usecols=use_features).values
X_total = np.concatenate([train_6, train_7, train_8, train_9], axis=0)
del train_6, train_7, train_8, train_9
gc.collect()

X_total = pd.DataFrame(X_total, columns=use_features)
N = X_total.shape[0]
N_cv = 25000000
Y_total = X_total["is_attributed"]
X_total = X_total.drop(["is_attributed"], axis=1)
X_train = X_total[:N-N_cv]
Y_train = Y_total[:N-N_cv]
X_cv = X_total[N-N_cv:]
Y_cv = Y_total[N-N_cv:]
del X_total, Y_total
gc.collect()
#Y_train = X_train['is_attributed'].values
#X_train = X_train.drop(["is_attributed"], axis=1).values
#print("Finished loading train!")

#gbm_train = lgbm.Dataset(X_train, Y_train, feature_name=use_features, categorical_feature=['app','device','os','channel','hour'])
#gbm_train.save_binary("../feature/train-8.bin")
#print("Finished save day 8 binary")

gbm_train = lgbm.Dataset(X_train, Y_train, categorical_feature=['app','device','os','channel','hour'])
gbm_train.save_binary("../feature/train-12fea.bin")
print("Finished loading train")

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

gbm_cv = gbm_train.create_valid(X_cv, Y_cv)
gbm_cv.save_binary("../feature/cv-12fea.bin")
print("Finished save cv binary")



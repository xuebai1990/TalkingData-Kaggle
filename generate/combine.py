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
    'p_click_ip_app_os':'uint32'
}

#use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel','click_os_channel','click_ip_app_os_channel_hour',\
#               'click_ip_app_os_device_hour','click_ip_app_os_device_minute','click_app_device','click_os_device','ins_minute_ave',\
#               'hour','click_device_channel','click_ip_app_os']
use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os']
use_base = ['app','device','os','channel','is_attributed']
use_base_test = ['app','device','os','channel']
use_group = ['total_click','click_per_channel','click_per_os','click_app_os','click_app_channel']
use_add = ['click_ip_app_os_device_hour','click_ip_app_os_device_minute']
use_add2 = ['hour','click_ip_app_os']
use_add3 = ['click_app','click_channel']
use_next = ['next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os']
use_next2 = ['next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel']
use_pf = ['p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os']

# Prepare train
train_base = pd.read_csv("../feature/train-day8.csv", dtype=dtypes, usecols=use_base)
train_group = pd.read_csv("../feature/train-day8-groupfeature.csv", dtype=dtypes, usecols=use_group)
train_add = pd.read_csv("../feature/train-day8-groupadd.csv", dtype=dtypes, usecols=use_add)
train_add2 = pd.read_csv("../feature/train-day8-group2.csv", dtype=dtypes, usecols=use_add2)
train_add3 = pd.read_csv("../feature/train-day8-group3.csv", dtype=dtypes, usecols=use_add3)
train_next = pd.read_csv("../feature/train-day8-nextclick.csv", dtype=dtypes, usecols=use_next)
train_next2 = pd.read_csv("../feature/train-day8-nextclick2.csv", dtype=dtypes, usecols=use_next2)
train_pf = pd.read_csv("../feature/train-day8-pfclick.csv", dtype=dtypes, usecols=use_pf)
X_train = pd.concat([train_base, train_group, train_add, train_add2, train_add3, train_next, train_next2, train_pf], axis=1)
X_train.to_csv("../feature/train-day8-total.csv", index=False)
del train_base, train_group, train_add, train_add2, train_add3, train_next, train_next2, train_pf, X_train
gc.collect()
print("Finished loading train!")

# Prepare validation
cv_base = pd.read_csv("../feature/train-day9.csv", dtype=dtypes, usecols=use_base)
cv_group = pd.read_csv("../feature/train-day9-groupfeature.csv", dtype=dtypes, usecols=use_group)
cv_add = pd.read_csv("../feature/train-day9-groupadd.csv", dtype=dtypes, usecols=use_add)
cv_add2 = pd.read_csv("../feature/train-day9-group2.csv", dtype=dtypes, usecols=use_add2)
cv_add3 = pd.read_csv("../feature/train-day9-group3.csv", dtype=dtypes, usecols=use_add3)
cv_next = pd.read_csv("../feature/train-day9-nextclick.csv", dtype=dtypes, usecols=use_next)
cv_next2 = pd.read_csv("../feature/train-day9-nextclick2.csv", dtype=dtypes, usecols=use_next2)
cv_pf = pd.read_csv("../feature/train-day9-pfclick.csv", dtype=dtypes, usecols=use_pf)
X_cv = pd.concat([cv_base, cv_group, cv_add, cv_add2, cv_add3, cv_next, cv_next2, cv_pf], axis=1)
X_cv.to_csv("../feature/train-day9-total.csv", index=False)
del cv_base, cv_group, cv_add, cv_add2, cv_add3, cv_next, cv_next2, cv_pf, X_cv
gc.collect()
print("Finished loading train!")

test_base = pd.read_csv("../feature/test.csv", dtype=dtypes, usecols=use_base_test)
test_group = pd.read_csv("../feature/test-groupfeature.csv", dtype=dtypes, usecols=use_group)
test_add = pd.read_csv("../feature/test-groupadd.csv", dtype=dtypes, usecols=use_add)
test_add2 = pd.read_csv("../feature/test-group2.csv", dtype=dtypes, usecols=use_add2)
test_add3 = pd.read_csv("../feature/test-group3.csv", dtype=dtypes, usecols=use_add3)
test_next = pd.read_csv("../feature/test-nextclick.csv", dtype=dtypes, usecols=use_next)
test_next2 = pd.read_csv("../feature/test-nextclick2.csv", dtype=dtypes, usecols=use_next2)
test_pf = pd.read_csv("../feature/test-pfclick.csv", dtype=dtypes, usecols=use_pf)
test = pd.concat([test_base, test_group, test_add, test_add2, test_add3, test_next, test_next2, test_pf], axis=1)
test.to_csv("../feature/test-total.csv", index=False)
del test_base, test_group, test_add, test_add2, test_add3, test_next, test_next2, test_pf, test
gc.collect()
print("Finished loading train!")



import pandas as pd
import numpy as np
import gc
import random

dtypes = {
    'click_id':'uint32',
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'is_attributed':'uint8'
}

use_col = ['ip', 'app', 'os', 'device', 'channel']

header = ['ip_uniq_app','ip_os_device_uniq_app','ip_uniq_device','ip_uniq_channel','ip_app_uniq_os','app_uniq_channel']

train1 = pd.read_csv('../feature/train-day6.csv', dtype=dtypes, usecols=use_col)
train2 = pd.read_csv('../feature/train-day7.csv', dtype=dtypes, usecols=use_col)
test = pd.read_csv('../feature/test.csv', dtype=dtypes, usecols=use_col)
dataset = [train1, train2, test]
del train1, train2, test
gc.collect()

dict = {1:'train-day6', 2:'train-day7', 3:'test'}

def countuniq(data, group, counted):
    agg_name = '_'.join(group)+'_uniq_'+counted
    uniq = data[group+[counted]].groupby(by=group)[counted].nunique().astype('uint32').reset_index().rename(columns={counted:agg_name})
    data = data.merge(uniq, on=group, how='left')
    del uniq
    gc.collect()
    return data

count = 0
for data in dataset:
    count += 1
    data = countuniq(data, ['ip'], 'app')
    data = countuniq(data, ['ip','os','device'], 'app')
    data = countuniq(data, ['ip'], 'device')
    data = countuniq(data, ['ip'], 'channel')
    data = countuniq(data, ['ip', 'app'], 'os')
    data = countuniq(data, ['app'], 'channel')
    data.to_csv("../feature/"+dict[count]+"-uniq.csv", columns = header, index=False)
    print("Finished!")

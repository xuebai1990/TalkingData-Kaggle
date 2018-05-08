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

use_col = ['ip', 'app', 'os', 'device', 'channel', 'click_time']

header = ['tar_ip', 'tar_app','tar_os','tar_device','tar_channel', 'tar_hour']

train = pd.read_csv('../feature/train-day8.csv', dtype=dtypes, usecols=use_col+['is_attributed'], parse_dates=['click_time'])
test = pd.read_csv('../feature/train-day9.csv', dtype=dtypes, usecols=use_col, parse_dates=['click_time'])
train['hour'] = train['click_time'].dt.hour.astype('uint8')
test['hour'] = test['click_time'].dt.hour.astype('uint8')
train = train.drop(['click_time'], axis=1)
test = test.drop(['click_time'], axis=1)

# Target encoding for ip
tar_ip = train[['ip','is_attributed']].groupby('ip', as_index=False).mean().astype('float32')
tar_ip.columns = ['ip', 'tar_ip']
train = train.merge(tar_ip, on='ip', how='left')
test = test.merge(tar_ip, on='ip', how='left')
print("Finish tar ip!")

# Target encoding for app
tar_app = train[['app','is_attributed']].groupby('app', as_index=False).mean().astype('float32')
tar_app.columns = ['app', 'tar_app']
train = train.merge(tar_app, on='app', how='left')
test = test.merge(tar_app, on='app', how='left')
print("Finish tar app!")

# Target encoding for os
tar_os = train[['os','is_attributed']].groupby('os', as_index=False).mean().astype('float32')
tar_os.columns = ['os', 'tar_os']
train = train.merge(tar_os, on='os', how='left')
test = test.merge(tar_os, on='os', how='left')
print("Finish tar os!")

# Target encoding for device
tar_device = train[['device','is_attributed']].groupby('device', as_index=False).mean().astype('float32')
tar_device.columns = ['device', 'tar_device']
train = train.merge(tar_device, on='device', how='left')
test = test.merge(tar_device, on='device', how='left')
print("Finish tar device!")

# Target encoding for channel
tar_channel = train[['channel','is_attributed']].groupby('channel', as_index=False).mean().astype('float32')
tar_channel.columns = ['channel', 'tar_channel']
train = train.merge(tar_channel, on='channel', how='left')
test = test.merge(tar_channel, on='channel', how='left')
print("Finish tar channel!")

# Target encoding for hour
tar_hour = train[['hour','is_attributed']].groupby('hour', as_index=False).mean().astype('float32')
tar_hour.columns = ['hour', 'tar_hour']
train = train.merge(tar_hour, on='hour', how='left')
test = test.merge(tar_hour, on='hour', how='left')
print("Finish tar hour!")

train.to_csv("../feature/train-day8-tarcode.csv", columns = header, index=False)
test.to_csv("../feature/train-day9-tarcode.csv", columns = header, index=False)

print("Finished!")

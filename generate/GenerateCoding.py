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

header = ['tar_app','tar_os','tar_device','tar_channel']

N_cv = 25000000

train = pd.read_csv('../feature/train.csv', dtype=dtypes, usecols=use_col+['is_attributed'], parse_dates=['click_time'])
test = pd.read_csv('../feature/test.csv', dtype=dtypes, usecols=use_col, parse_dates=['click_time'])
X = train[:train.shape[0]-N_cv]

# Target encoding for app
tar_app = X[['app','is_attributed']].groupby('app', as_index=False).mean().astype('float32')
tar_app.columns = ['app', 'tar_app']
train = train.merge(tar_app, on='app', how='left')
test = test.merge(tar_app, on='app', how='left')
print("Finish tar app!")

# Target encoding for os
tar_os = X[['os','is_attributed']].groupby('os', as_index=False).mean().astype('float32')
tar_os.columns = ['os', 'tar_os']
train = train.merge(tar_os, on='os', how='left')
test = test.merge(tar_os, on='os', how='left')
print("Finish tar os!")

# Target encoding for device
tar_device = X[['device','is_attributed']].groupby('device', as_index=False).mean().astype('float32')
tar_device.columns = ['device', 'tar_device']
train = train.merge(tar_device, on='device', how='left')
test = test.merge(tar_device, on='device', how='left')
print("Finish tar device!")

# Target encoding for channel
tar_channel = X[['channel','is_attributed']].groupby('channel', as_index=False).mean().astype('float32')
tar_channel.columns = ['channel', 'tar_channel']
train = train.merge(tar_channel, on='channel', how='left')
test = test.merge(tar_channel, on='channel', how='left')
print("Finish tar channel!")

train.to_csv("../feature/train-tarcode.csv", columns = header, index=False)
test.to_csv("../feature/test-tarcode.csv", columns = header, index=False)

print("Finished!")

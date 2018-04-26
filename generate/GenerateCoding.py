import pandas as pd
import numpy as np
import gc
import random

dtypes = {
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'is_attributed':'uint8'
}

dtypes_test = {
    'click_id':'uint32',
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16'
}

header = ['tar_app','tar_os','tar_device','tar_channel']

day8 = pd.read_csv('../feature/train-day8.csv', dtype=dtypes)
day9 = pd.read_csv('../feature/train-day9.csv', dtype=dtypes)
day8 = day8.drop(["attributed_time"], axis=1)
day9 = day9.drop(["attributed_time"], axis=1)

# Target encoding for app
tar_app = day8[['app','is_attributed']].groupby('app', as_index=False).mean().astype('float32')
tar_app.columns = ['app', 'tar_app']
day8 = day8.merge(tar_app, on='app', how='left')
day9 = day9.merge(tar_app, on='app', how='left')
print("Finish tar app!")

# Target encoding for os
tar_os = day8[['os','is_attributed']].groupby('os', as_index=False).mean().astype('float32')
tar_os.columns = ['os', 'tar_os']
day8 = day8.merge(tar_os, on='os', how='left')
day9 = day9.merge(tar_os, on='os', how='left')
print("Finish tar os!")

# Target encoding for device
tar_device = day8[['device','is_attributed']].groupby('device', as_index=False).mean().astype('float32')
tar_device.columns = ['device', 'tar_device']
day8 = day8.merge(tar_device, on='device', how='left')
day9 = day9.merge(tar_device, on='device', how='left')
print("Finish tar device!")

# Target encoding for channel
tar_channel = day8[['channel','is_attributed']].groupby('channel', as_index=False).mean().astype('float32')
tar_channel.columns = ['channel', 'tar_channel']
day8 = day8.merge(tar_channel, on='channel', how='left')
day9 = day9.merge(tar_channel, on='channel', how='left')
print("Finish tar channel!")

day8.to_csv("../feature/train-day8-tarcode.csv", columns = header, index=False)
day9.to_csv("../feature/train-day9-tarcode.csv", columns = header, index=False)

print("Finished!")

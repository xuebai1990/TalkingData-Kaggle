import pandas as pd
import numpy as np
import gc
import random

dtypes_train = {
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'is_attributed':'uint8'
}

#train_8 = pd.read_csv('../feature/train-day8.csv', dtype=dtypes, parse_dates=['click_time'])
#train_9 = pd.read_csv('../feature/train-day9.csv', dtype=dtypes, parse_dates=['click_time'])

dtypes_test = {
    'click_id':'uint32',
    'ip':'uint32',
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16'
}

#test = pd.read_csv('../feature/test.csv', dtype=dtypes, parse_dates=['click_time'])

# Drop some unimportant information
#test = test.drop(['click_id'], axis=1)
#train_8 = train_8.drop(['attributed_time'], axis=1)
#train_9 = train_9.drop(['attributed_time'], axis=1)

total = ["train-day8","train-day9","test"]
#total = ["test"]
header = ['click_app', 'click_os', 'click_device', 'click_channel']

for key in total:

    if key == "test":
        dtypes = dtypes_test
    else:
        dtypes = dtypes_train

    df = pd.read_csv('../feature/'+key+'.csv', dtype=dtypes, parse_dates=['click_time'])
    if "click_id" in df.columns:
        df = df.drop(['click_id'], axis=1)
        df['is_attributed'] = np.zeros(df.shape[0]).astype('uint8')
    if "attributed_time" in df.columns:
        df = df.drop(["attributed_time"], axis=1)

    # Extract day hour info
#    df['day'] = df['click_time'].dt.day.astype('uint8')
#    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df = df.drop(['click_time'], axis=1)

    # Total number of click per app
    num_click_per_ip = df[['app','is_attributed']].groupby(by=['app']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['app','click_app']
    df = df.merge(num_click_per_ip, on=['app'], how='left')
    print("Finish app!")

    # Total number of click per os
    num_click_per_ip = df[['os','is_attributed']].groupby(by=['os']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['os','click_os']
    df = df.merge(num_click_per_ip, on=['os'], how='left')
    print("Finish os!")

    # Total number of click per device
    num_click_per_ip = df[['device','is_attributed']].groupby(by=['device']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['device','click_device']
    df = df.merge(num_click_per_ip, on=['device'], how='left')
    print("Finish device!")

    # Total number of click per channel
    num_click_per_ip = df[['channel','is_attributed']].groupby(by=['channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['channel','click_channel']
    df = df.merge(num_click_per_ip, on=['channel'], how='left')
    print("Finish channel!")

    df.to_csv("../feature/"+key+"-group3.csv", columns = header, index=False)
    del df
    gc.collect()

    print("Finished!")

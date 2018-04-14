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
header = ['next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os']

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

    # Next click each ip
    df['next_click_ip'] = df[['ip','click_time']].groupby(by=['ip']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Next click each (ip, channel)
    df['next_click_ip_channel'] = df[['ip','channel','click_time']].groupby(by=['ip','channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Next click each (ip, app)
    df['next_click_ip_app'] = df[['ip','app','click_time']].groupby(by=['ip','app']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Next click each (ip, device)
    df['next_click_ip_device'] = df[['ip','device','click_time']].groupby(by=['ip','device']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Next click each (ip, os)
    df['next_click_ip_os'] = df[['ip','os','click_time']].groupby(by=['ip','os']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    df.to_csv("../feature/"+key+"-nextclick.csv", columns = header, index=False)
    del df
    gc.collect()

    print("Finished!")

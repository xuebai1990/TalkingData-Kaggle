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
header = ['p_click_ip','p_click_ip_channel','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os']

for key in total:

    if key == "test":
        dtypes = dtypes_test
    else:
        dtypes = dtypes_train

    df = pd.read_csv('../feature/'+key+'.csv', dtype=dtypes)
    if "click_id" in df.columns:
        df = df.drop(['click_id','click_time'], axis=1)
        df['is_attributed'] = np.zeros(df.shape[0]).astype('uint8')
    if "attributed_time" in df.columns:
        df = df.drop(['attributed_time','click_time'], axis=1)

    # Previous/future total click per ip
    df['p_click_ip'] = df.groupby(by=['ip']).cumcount()

    # Previous/future click each (ip, channel)
    df['p_click_ip_channel'] = df.groupby(by=['ip','channel']).cumcount()

    # Previous/future click each (ip, app)
    df['p_click_ip_app'] = df.groupby(by=['ip','app']).cumcount()

    # Previous/future click each (ip, device)
    df['p_click_ip_device'] = df.groupby(by=['ip','device']).cumcount()

    # Previous/future click each (ip, os)
    df['p_click_ip_os'] = df.groupby(by=['ip','os']).cumcount()

    # Previous click each (ip, app, os, device)
    df['p_click_ip_app_os_device'] = df.groupby(by=['ip','app','os','device']).cumcount()

    # Previous click each (ip, app, os)
    df['p_click_ip_app_os'] = df.groupby(by=['ip','app','os']).cumcount()

    df.to_csv("../feature/"+key+"-pfclick.csv", columns = header, index=False)
    del df
    gc.collect()

    print("Finished!")

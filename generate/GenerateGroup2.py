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
header = ['hour','click_device_channel','click_ip_app_hour','click_ip_os_hour','click_ip_device_hour','click_ip_channel_hour',\
         'click_ip_app_os']

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
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df = df.drop(['click_time'], axis=1)

    # Total number of click per ip
    num_click_per_ip = df[['device','channel','is_attributed']].groupby(by=['device','channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['device', 'channel', 'click_device_channel']
    df = df.merge(num_click_per_ip, on=['device','channel'], how='left')
    print("Finish device channel!")

    num_click_per_ip = df[['ip','app','hour','is_attributed']].groupby(by=['ip','app','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip','app','hour','click_ip_app_hour']
    df = df.merge(num_click_per_ip, on=['ip','app','hour'], how='left')
    print("Finish ip app hour!")

    num_click_per_ip = df[['ip','os','hour','is_attributed']].groupby(by=['ip','os','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip','os','hour','click_ip_os_hour']
    df = df.merge(num_click_per_ip, on=['ip','os','hour'], how='left')
    print("Finish ip os hour!")

    num_click_per_ip = df[['ip','device','hour','is_attributed']].groupby(by=['ip','device','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip','device','hour','click_ip_device_hour']
    df = df.merge(num_click_per_ip, on=['ip','device','hour'], how='left')
    print("Finish ip device hour!")

    num_click_per_ip = df[['ip','channel','hour','is_attributed']].groupby(by=['ip','channel','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip','channel','hour','click_ip_channel_hour']
    df = df.merge(num_click_per_ip, on=['ip','channel','hour'], how='left')
    print("Finish ip channel hour!")

    num_click_per_ip = df[['ip','app','os','is_attributed']].groupby(by=['ip','app','os']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip','app','os','click_ip_app_os']
    df = df.merge(num_click_per_ip, on=['ip','app','os'], how='left')
    print("Finish ip app os!")

    df.to_csv("../feature/"+key+"-group2.csv", columns = header, index=False)
    del df
    gc.collect()

    print("Finished!")

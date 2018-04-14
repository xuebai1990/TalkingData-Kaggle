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
header = ['total_click','click_per_hour','click_per_channel','click_per_app','click_per_device','click_per_os',\
          'click_app_os','click_app_channel','click_os_channel','click_ip_app_os_channel_hour']

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
    num_click_per_ip = df[['ip','is_attributed']].groupby('ip', as_index=False).count().astype('uint32')
    num_click_per_ip.columns = ['ip', 'total_click']

    df = df.merge(num_click_per_ip, on='ip', how='left')
    print("Finish total click!")

    # Total number of click per hour per ip
    num_click_per_ip = df[['ip','hour','is_attributed']]\
                      .groupby(by=['ip','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'hour', 'click_per_hour']

    df = df.merge(num_click_per_ip, on=['ip', 'hour'], how='left')
    print("Finish ip hour!")

    # Total number of click per channel per ip
    num_click_per_ip = df[['ip','channel','is_attributed']]\
                       .groupby(by=['ip','channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'channel', 'click_per_channel']

    df = df.merge(num_click_per_ip, on=['ip', 'channel'], how='left')
    print("Finish ip channel!")

    # Total number of click per app per ip
    num_click_per_ip = df[['ip','app','is_attributed']]\
                       .groupby(by=['ip','app']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'app', 'click_per_app']

    df = df.merge(num_click_per_ip, on=['ip', 'app'], how='left')
    print("Finish ip app!")

    # Total number of click per device per ip
    num_click_per_ip = df[['ip','device','is_attributed']]\
                       .groupby(by=['ip','device']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'device', 'click_per_device']

    df = df.merge(num_click_per_ip, on=['ip', 'device'], how='left')
    print("Finish ip device!")

    # Total number of click per os per ip
    num_click_per_ip = df[['ip','os','is_attributed']]\
                       .groupby(by=['ip','os']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'os', 'click_per_os']

    df = df.merge(num_click_per_ip, on=['ip', 'os'], how='left')
    print("Finish ip os!")

    # Total number of click per app per os
    num_click_per_ip = df[['app','os','is_attributed']]\
                       .groupby(by=['app','os']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['app', 'os', 'click_app_os']

    df = df.merge(num_click_per_ip, on=['app', 'os'], how='left')
    print("Finish app os!")

    # Total number of click per app per channel
    num_click_per_ip = df[['app','channel','is_attributed']]\
                       .groupby(by=['app','channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['app', 'channel', 'click_app_channel']

    df = df.merge(num_click_per_ip, on=['app', 'channel'], how='left')
    print("Finish ip os!")

    # Total number of click per os per channel
    num_click_per_ip = df[['os','channel','is_attributed']]\
                       .groupby(by=['os','channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['os', 'channel', 'click_os_channel']

    df = df.merge(num_click_per_ip, on=['os', 'channel'], how='left')
    print("Finish os channel!")

    # Total number of click per (ip, app, os, channel, hour)
    num_click_per_ip = df[['ip','app','os','channel','hour','is_attributed']]\
                       .groupby(by=['ip','app','os','channel','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'app','os','channel','hour','click_ip_app_os_channel_hour']

    df = df.merge(num_click_per_ip, on=['ip','app','os','channel','hour'], how='left')
    print("Finish ip app os channel hour!")

    df.to_csv("../feature/"+key+"-groupfeature.csv", columns = header, index=False)
    del df
    gc.collect()

    print("Finished!")

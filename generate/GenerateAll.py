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

total = ["train-day7","train-day6"]
#total = ["test"]
use_features = ['app','device','os','channel','is_attributed','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ipp_app_os']


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
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
#    df = df.drop(['click_time'], axis=1)

    # Total number of click per ip
    num_click_per_ip = df[['ip','is_attributed']].groupby('ip', as_index=False).count().astype('uint32')
    num_click_per_ip.columns = ['ip', 'total_click']
    df = df.merge(num_click_per_ip, on='ip', how='left')
    print("Finish total click!")

    # Total number of click per channel per ip
    num_click_per_ip = df[['ip','channel','is_attributed']]\
                       .groupby(by=['ip','channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'channel', 'click_per_channel']
    df = df.merge(num_click_per_ip, on=['ip', 'channel'], how='left')
    print("Finish ip channel!")


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

    # Total number of click per (ip, app, os, device, hour)
    num_click_per_ip = df[['ip','app','os','device','hour','is_attributed']]\
                       .groupby(by=['ip','app','os','device','hour']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'app','os','device','hour','click_ip_app_os_device_hour']
    df = df.merge(num_click_per_ip, on=['ip','app','os','device','hour'], how='left')
    print("Finish ip app os device hour!")

    # Total number of click per (ip, app, os, device, minute)
    num_click_per_ip = df[['ip','app','os','device','hour','minute','is_attributed']]\
                       .groupby(by=['ip','app','os','device','hour','minute']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip', 'app','os','device','hour','minute','click_ip_app_os_device_minute']
    df = df.merge(num_click_per_ip, on=['ip','app','os','device','hour','minute'], how='left')
    print("Finish ip app os device minute!")

    # Click ip app os
    num_click_per_ip = df[['ip','app','os','is_attributed']].groupby(by=['ip','app','os']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['ip','app','os','click_ip_app_os']
    df = df.merge(num_click_per_ip, on=['ip','app','os'], how='left')
    print("Finish ip app os!")

    # Total number of click per app
    num_click_per_ip = df[['app','is_attributed']].groupby(by=['app']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['app','click_app']
    df = df.merge(num_click_per_ip, on=['app'], how='left')
    print("Finish app!")

    # Total number of click per channel
    num_click_per_ip = df[['channel','is_attributed']].groupby(by=['channel']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['channel','click_channel']
    df = df.merge(num_click_per_ip, on=['channel'], how='left')
    print("Finish channel!")

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

    # Next click each ip, app, os, device
    df['next_click_ip_app_os_device'] = df[['ip','app','os','device','click_time']]\
                                        .groupby(by=['ip','app','os','device']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Next click each ip, app, os
    df['next_click_ip_app_os'] = df[['ip','app','os','click_time']]\
                                        .groupby(by=['ip','app','os']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Next click each app, channel
    df['next_click_app_channel'] = df[['app','channel','click_time']]\
                                        .groupby(by=['app','channel']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds

    # Previous/future total click per ip
    df['p_click_ip'] = df.groupby(by=['ip']).cumcount()

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


    df.to_csv("../feature/"+key+"-total.csv", columns = use_features, index=False)
    del df
    gc.collect()

    print("Finished!")

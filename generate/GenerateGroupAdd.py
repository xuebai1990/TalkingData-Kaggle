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
header = ['click_ip_app_os_device_hour','click_ip_app_os_device_minute','click_app_device','click_os_device',\
         'ip_minute_ave','ip_minute_std','ins_minute_ave','ins_minute_std']

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
    df = df.drop(['click_time'], axis=1)

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

    # Total number of click per app per device
    num_click_per_ip = df[['app','device','is_attributed']]\
                       .groupby(by=['app','device']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['app', 'device', 'click_app_device']
    df = df.merge(num_click_per_ip, on=['app', 'device'], how='left')
    print("Finish app device!")   

    # Total number of click per os per device
    num_click_per_ip = df[['os','device','is_attributed']]\
                       .groupby(by=['os','device']).count().astype('uint32').reset_index()
    num_click_per_ip.columns = ['os', 'device', 'click_os_device']
    df = df.merge(num_click_per_ip, on=['os', 'device'], how='left')
    print("Finish os device!")
    del num_click_per_ip
    gc.collect()

    # IP ave on minute click and std
    grouped = df[['ip','hour','minute','is_attributed']].groupby(by=['ip','hour','minute']).count().astype('uint32').\
          reset_index()
    gg = grouped[['ip','is_attributed']].groupby(by=['ip'])['is_attributed'].agg([np.average, np.std]).reset_index()
    gg.columns = ['ip','ip_minute_ave','ip_minute_std']
    df = df.merge(gg, on=['ip'], how='left')
    print("Finished ip ave std")

    # IP app os device on minute click and std
    grouped = df[['ip','app','os','device','hour','minute','is_attributed']]\
             .groupby(by=['ip','app','os','device','hour','minute']).count().astype('uint32').reset_index()
    gg = grouped[['ip','app','os','device','is_attributed']].groupby(by=['ip','app','os','device'])['is_attributed'].agg([np.average, np.std]).reset_index()
    gg.columns = ['ip','app','os','device','ins_minute_ave','ins_minute_std']
    df = df.merge(gg, on=['ip','app','os','device'], how='left')
    print("Finished ins ave std")  
    del grouped, gg
    gc.collect()

    df.to_csv("../feature/"+key+"-groupadd.csv", columns = header, index=False)
    del df
    gc.collect()

    print("Finished!")

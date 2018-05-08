import pandas as pd
import numpy as np
import gc
from dtypes import dtypes
from sklearn.datasets import dump_svmlight_file

def fillna(data, column, val):
    data[column].fillna(value=val, inplace=True)

def standard(data, column):
    data[column] = (data[column] - data[column].mean())/data[column].std()

category = ['app','os','device','channel','hour']
category_index = [0, 1, 2, 3, 11]

X_day8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes)
X_day9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes)
X_test = pd.read_csv("../feature/test-total.csv", dtype=dtypes)

N8 = X_day8.shape[0]
N9 = X_day9.shape[0]
NT = X_test.shape[0]

for data in (X_day8, X_day9, X_test):
    for column in ('next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
                   'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel'):
        fillna(data, column, data[column].max())
    for column in ('p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os'):
        fillna(data, column, 0)
del data
gc.collect()
print("Finished loading data!")
X_total = pd.concat([X_day8, X_day9], axis=0, ignore_index=True)
del X_day8, X_day9
gc.collect()

Y_total = X_total["is_attributed"]
X_total = X_total.drop(["is_attributed"], axis=1)

X_total = pd.concat([X_total, X_test], axis=0, ignore_index=True)
del X_test
gc.collect()

start = {}
end = {}
min_val = {}
max_val = {}
index = 0
count = 0
for column in X_total.columns:
    if column in category:
        start[count] = index
        min_ind = X_total[column].min().astype('uint16')
        min_val[count] = min_ind
        max_ind = X_total[column].max().astype('uint16')
        end[count] = index + max_ind - min_ind
        index = end[count] + 1
    else:
        start[count] = end[count] = index
        min_val[count] = X_total[column].min()
        max_val[count] = X_total[column].max()
        index += 1
    count += 1

Y_total = Y_total.values

fout = open("day8.dat", 'w')
X_day8 = X_total[:N8].values
for i in range(N8):
    line = []
    if Y_total[i] > 0.5:
        label = "1"
    else:
        label = "0"
    line.append(label)
    for j in range(X_day8.shape[1]):
        if np.absolute(X_day8[i][j]) < 1e-8:
            continue
        if j in category_index:
            field = str(int(X_day8[i][j] - min_val[j] + start[j]))
            value = "1"
            item = "{}:{}".format(field, value)
        else:
            field = str(start[j])
            value = (X_day8[i][j] - min_val[j]) / (max_val[j] - min_val[j])
            if np.absolute(value) < 1e-8:
                continue
            item = "{}:{:8.6f}".format(field, value)
        line.append(item)
    line.append("\n")
    fout.write(" ".join(line))
del X_day8
gc.collect()
fout.close()
print("Finished write day8")

fout = open("day9.dat", 'w')
X_day9 = X_total[N8:N8+N9].values
for i in range(N9):
    line = []
    if Y_total[i+N8] > 0.5:
        label = "1"
    else:
        label = "0"
    line.append(label)
    for j in range(X_day9.shape[1]):
        if np.absolute(X_day9[i][j]) < 1e-8:
            continue
        if j in category_index:
            field = str(int(X_day9[i][j] - min_val[j] + start[j]))
            value = "1"
            item = "{}:{}".format(field, value)
        else:
            field = str(start[j])
            value = (X_day9[i][j] - min_val[j]) / (max_val[j] - min_val[j])
            if np.absolute(value) < 1e-8:
                continue
            item = "{}:{:8.6f}".format(field, value)
        line.append(item)
    line.append("\n")
    fout.write(" ".join(line))
del X_day9
gc.collect()
fout.close()
print("Finished write day9")

fout = open("test-sparse.dat", 'w')
X_test = X_total[N8+N9:].values
for i in range(NT):
    line = []
    label = "1"
    line.append(label)
    for j in range(X_test.shape[1]):
        if np.absolute(X_test[i][j]) < 1e-8:
            continue
        if j in category_index:
            field = str(int(X_test[i][j] - min_val[j] + start[j]))
            value = "1"
            item = "{}:{}".format(field, value)
        else:
            field = str(start[j])
            value = (X_test[i][j] - min_val[j]) / (max_val[j] - min_val[j])
            if np.absolute(value) < 1e-8:
                continue
            item = "{}:{:8.6f}".format(field, value)
        line.append(item)
    line.append("\n")
    fout.write(" ".join(line))
fout.close()
print("Finished write test")

            


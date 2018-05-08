import numpy as np
import pandas as pd
import lightgbm as lgbm
from dtypes import dtypes
import gc

bst = lgbm.Booster(model_file='model.txt')

X_day8 = pd.read_csv("../feature/train-day8-total.csv", dtype=dtypes)
Y_day8 = X_day8["is_attributed"]
X_day8 = X_day8.drop(["is_attributed"], axis=1)
train_pred = bst.predict(X_day8)
train_frame = pd.DataFrame({'lightgbm':train_pred, 'label':Y_day8.values})
train_frame.to_csv('day8-lightgbm.csv', index=False)
del X_day8, Y_day8
gc.collect()

X_day9 = pd.read_csv("../feature/train-day9-total.csv", dtype=dtypes)
Y_day9 = X_day9["is_attributed"]
X_day9 = X_day9.drop(["is_attributed"], axis=1)
cv_pred = bst.predict(X_day9)
cv_frame = pd.DataFrame({'lightgbm':cv_pred, 'label':Y_day9.values})
cv_frame.to_csv('day9-lightgbm.csv', index=False)
del X_day9, Y_day9
gc.collect()

X_test = pd.read_csv("../feature/test-total.csv", dtype=dtypes)
test_pred = bst.predict(X_test)
test_frame = pd.DataFrame({'lightgbm':test_pred})
test_frame.to_csv('test-lightgbm.csv', index=False)

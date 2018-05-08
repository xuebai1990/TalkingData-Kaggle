import numpy as np
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file

X_day8, Y_day8 = load_svmlight_file("day8.dat", dtype=np.float16)
print("Finished loading data!")


rf = RandomForestClassifier(n_estimators=200,
                   min_samples_leaf=5000,
                   max_depth=31,
                   n_jobs=16,
                   random_state=0,
                   class_weight={0:1,1:200})

rf.fit(X_day8, Y_day8)
print("Finished training!")

train_pred = rf.predict_proba(X_day8)[:,1]
print("Training auc: {}".format(roc_auc_score(Y_day8, train_pred)))
del X_day8, Y_day8
gc.collect()

X_day9, Y_day9 = load_svmlight_file("day9.dat", dtype=np.float32)
cv_pred = rf.predict_proba(X_day9)[:,1]
print("Test auc: {}".format(roc_auc_score(Y_day9, cv_pred)))
train_frame = pd.DataFrame({'rf':train_pred})
train_frame.to_csv('day8-rf.csv', index=False)
cv_frame = pd.DataFrame({'rf':cv_pred})
cv_frame.to_csv('day9-rf.csv', index=False)


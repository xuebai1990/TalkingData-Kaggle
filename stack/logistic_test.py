import numpy as np
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.datasets import load_svmlight_file

X_day8, Y_day8 = load_svmlight_file("day8.dat", dtype=np.float16)
print("Finished loading data!")

#lg = LogisticRegression(n_jobs=16,
#                        random_state=0,
#                        class_weight={0:1,1:200},
#                        C = 5.0,
#                        solver='saga')

lg = SGDClassifier(loss='log',
                   penalty='l2',
                   n_jobs=16,
                   random_state=0,
                   class_weight={0:1,1:200},
                   max_iter=20,
                   tol=1e-4
                   )

lg.fit(X_day8, Y_day8)
print("Finished training!")

train_pred = lg.predict_proba(X_day8)[:,1]
print("Training auc: {}".format(roc_auc_score(Y_day8, train_pred)))
del X_day8, Y_day8
gc.collect()

X_day9, Y_day9 = load_svmlight_file("day9.dat", dtype=np.float16)
cv_pred = lg.predict_proba(X_day9)[:,1]
print("Test auc: {}".format(roc_auc_score(Y_day9, cv_pred)))
train_frame = pd.DataFrame({'lg':train_pred})
train_frame.to_csv('day8-lg.csv', index=False)
cv_frame = pd.DataFrame({'lg':cv_pred})
cv_frame.to_csv('day9-lg.csv', index=False)
del X_day9, Y_day9
gc.collect()

X_test, Y_test = load_svmlight_file("test-sparse.dat", dtype=np.float16)
test_pred = lg.predict_proba(X_test)[:,1]
test_frame = pd.DataFrame({'lg':test_pred})
test_frame.to_csv('test-lg.csv', index=False)

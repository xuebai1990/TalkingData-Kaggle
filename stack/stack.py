import numpy as np
import pandas as pd
import lightgbm as lgbm
import gc
from sklearn.metrics import roc_auc_score

lgbm8 = pd.read_csv("result/day8-lightgbm.csv")
Y_train = lgbm8["label"]
lgbm8 = lgbm8.drop(["label"], axis=1)
rf8 = pd.read_csv("result/day8-rf.csv")
lg8 = pd.read_csv("result/day8-lg.csv")

X_train = pd.concat([lgbm8, rf8, lg8], axis=1)
del lgbm8, rf8, lg8
gc.collect()

lgbm9 = pd.read_csv("result/day9-lightgbm.csv")
Y_cv = lgbm9["label"]
lgbm9 = lgbm9.drop(["label"], axis=1)
rf9 = pd.read_csv("result/day9-rf.csv")
lg9 = pd.read_csv("result/day9-lg.csv")
print(roc_auc_score(Y_cv, lgbm9))

X_cv = pd.concat([lgbm9, rf9, lg9], axis=1)
del lgbm9, rf9, lg9
gc.collect()

gbm_train = lgbm.Dataset(X_train, Y_train)
gbm_cv = lgbm.Dataset(X_cv, Y_cv)

params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'gbdt',
        
        # Deal with overfitting
        'bagging_fraction': 0.9, 
        'bagging_freq': 1,
        'min_data_in_leaf': 5000,
        'feature_fraction': 1.0,
        'num_leaves': 11,
        'max_depth': -1,
        'max_bin': 255,
        
        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 300,
}

bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)

click_id = np.loadtxt("../feature/test_click_id.csv").astype(np.uint32)
lgbm_test = pd.read_csv("result/test-lightgbm.csv")
rf_test = pd.read_csv("result/test-rf.csv")
lg_test = pd.read_csv("result/test-lg.csv")

X_test = pd.concat([lgbm_test, rf_test, lg_test], axis=1)
del lgbm_test, rf_test, lg_test
gc.collect()

test_pred = bst.predict(X_test)

submit = pd.DataFrame({'click_id':click_id, 'is_attributed':test_pred})
submit.to_csv('submission.csv', index=False)

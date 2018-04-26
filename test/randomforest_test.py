import numpy as np
import pandas as pd
import lightgbm as lgbm


use_features = ['app','device','os','channel','total_click','click_per_channel','click_per_os','click_app_os','click_app_channel',\
               'click_ip_app_os_device_hour','click_ip_app_os_device_minute',\
               'hour','click_ip_app_os','click_app','click_channel',\
               'next_click_ip','next_click_ip_channel','next_click_ip_app','next_click_ip_device','next_click_ip_os',\
               'next_click_ip_app_os_device','next_click_ip_app_os','next_click_app_channel',\
               'p_click_ip','p_click_ip_app','p_click_ip_device','p_click_ip_os','p_click_ip_app_os_device','p_click_ip_app_os']

gbm_train = lgbm.Dataset("../feature/train-total.bin", feature_name=use_features,\
                 categorical_feature=['app','device','os','channel','hour'])
gbm_cv = lgbm.Dataset("../feature/cv-total.bin", feature_name=use_features,\
                categorical_feature=['app','device','os','channel','hour'])
    
params = {
        # Task based parameter
        'application' :'binary',
        'learning_rate' : 0.1,
        'num_iterations': 1000,
        'boosting' : 'rf',
        
        # Deal with overfitting
        'bagging_fraction': 0.9, 
        'bagging_freq': 1,
        'min_data_in_leaf': 5000,
        'feature_fraction': 0.2,
        'num_leaves': 31,
        'max_depth': -1,
        'max_bin': 255,
        
        # Others
        'metric': 'auc',
        'num_threads': 16,
        'scale_pos_weight': 200,
#        'ignore_column':'name:p_click_ip_app_os'
}

bst = lgbm.train(params, gbm_train, valid_sets=[gbm_cv], early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)

# Predict
click_id = np.loadtxt("../feature/test_click_id.csv").astype(np.uint32)
test = pd.read_csv("../feature/test-total.csv")

test_pred = bst.predict(test, num_iteration=bst.best_iteration)

submit = pd.DataFrame({'click_id':click_id, 'is_attributed':test_pred})
submit.to_csv('submission.csv', index=False)


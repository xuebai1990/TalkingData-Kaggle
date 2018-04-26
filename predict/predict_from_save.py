import numpy as np
import pandas as pd
import lightgbm as lgbm

model = lgbm.Booster(model_file='model.txt')

#use_features = ['app','device','os','channel','hour','total_click','next_click_ip_app_os_device','p_click_ip']

# Predict
click_id = np.loadtxt("../feature/test_click_id.csv").astype(np.uint32)
test = pd.read_csv("../feature/test-total.csv")

test_pred = model.predict(test)

submit = pd.DataFrame({'click_id':click_id, 'is_attributed':test_pred})
submit.to_csv('submission.csv', index=False)


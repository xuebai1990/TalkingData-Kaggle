import numpy as np
import pandas as pd

lgbm = pd.read_csv("../submission/lightgbm/lightgbm-29feature-alltrain-25mvalid.csv")
rf = pd.read_csv("../submission/randomforest/randomforest-29features-alltrain-25mvalid.csv")

click_id = lgbm["click_id"]
test_pred = (lgbm["is_attributed"] + rf["is_attributed"]) / 2.0

submit = pd.DataFrame({'click_id':click_id, 'is_attributed':test_pred})
submit.to_csv('submission.csv', index=False)

import pandas as pd
import numpy as np
import gc

dtypes = {
    'app':'uint16',
    'device':'uint16',
    'os':'uint16',
    'channel':'uint16',
    'is_attributed':'uint8',
    'hour':'uint8',
    'total_click':'uint32',
    'click_per_hour':'uint32',
    'click_per_channel':'uint32',
    'click_per_app':'uint32',
    'click_per_device':'uint32',
    'click_per_os':'uint32'   
}


# Prepare train, test
train = pd.read_csv("train_feature.csv", dtype=dtypes)
test = pd.read_csv("test_feature.csv", dtype=dtypes)

# Turn categorical features into dummy variables
train = pd.get_dummies(train, columns=['app','device','os','channel','hour'])
test = pd.get_dummies(test, columns=['app','device','os','channel','hour'])

X_train = train.drop(['is_attributed'], axis=1).values
Y_train = train['is_attributed'].values
X_test = test.drop(['is_attributed'], axis=1).values
click_id = np.loadtxt("test_click_id.csv").astype(np.uint32)

del train, test
gc.collect()
print("Finshed loading data!")


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
result = np.zeros(X_test.shape[0])
train_result = np.zeros(X_train.shape[0])


lg = LogisticRegression(n_jobs=8,
                        random_state=0,
                        class_weight={0:1,1:200})

for (train_index, cv_index) in sss.split(X_train, Y_train):

    XX_train, YY_train = X_train[train_index], Y_train[train_index]
    XX_cv, YY_cv = X_train[cv_index], Y_train[cv_index]
    
    lg.fit(XX_train, YY_train)
    ypred = lg.predict_proba(X_test)
    result += ypred

    train_pred = lg.predict_proba(XX_cv)
    train_result[cv_index] = train_pred
    print(roc_auc_score(YY_cv, train_pred))
    

result /= 5


submit = pd.DataFrame({'click_id':click_id, 'is_attributed':result})
submit.to_csv('submit.csv', index=False)
train_result = pd.DataFrame({"LogisticRegression":train_result})
train_result.to_csv("train_predict.csv", index=False)


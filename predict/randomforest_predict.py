import pandas as pd
import numpy as np
import gc
from pandas.core.categorical import Categorical
from scipy.sparse import csr_matrix, hstack

def sparse_dummies(df, column):
    """Returns sparse OHE matrix for the column of the dataframe"""
    categories = Categorical(df[column])
    column_names = np.array([f"{column}_{str(i)}" for i in range(len(categories.categories))])
    N = len(categories)
    row_numbers = np.arange(N, dtype=np.int)
    ones = np.ones((N,))
    return csr_matrix((ones, (row_numbers, categories.codes))), column_names

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


# Prepare train
train = pd.read_csv("train_feature.csv", dtype=dtypes, skiprows=range(1,35000000))
test = pd.read_csv("test_feature.csv", dtype=dtypes)

N = train.shape[0]
Y_train = train['is_attributed'].values
total = pd.concat([train, test], axis=0, ignore_index=True)
del train, test
gc.collect()

# Turn categorical features into dummy variables
categorical_features = ['app', 'device', 'os', 'channel','hour']
numerical_features = ['total_click','click_per_hour','click_per_channel','click_per_app','click_per_device','click_per_os']
matrices = []
all_column_names = []
# creates a matrix per categorical feature
for c in categorical_features:
    matrix, column_names = sparse_dummies(total, c)
    matrices.append(matrix)
    all_column_names.append(column_names)

# appends a matrix for numerical features (one column per feature)
matrices.append(csr_matrix(total[numerical_features].values, dtype=float))
all_column_names.append(total[numerical_features].columns.values)

total_sparse = hstack(matrices, format="csr")
feature_names = np.concatenate(all_column_names)
del matrices, all_column_names

X_train = total_sparse[:N]
X_test = total_sparse[N:]

del total
gc.collect()

click_id = np.loadtxt("test_click_id.csv").astype(np.uint32)

print("Finshed all data processing!")


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
result = np.zeros(X_test.shape[0])
train_result = np.zeros(X_train.shape[0])


rf = RandomForestClassifier(n_estimators=100, 
                            max_depth=5,
                            min_samples_leaf=100,
                            oob_score=False,
                            n_jobs=8,
                            random_state=0,
                            class_weight={0:1,1:200})

for (train_index, cv_index) in sss.split(X_train, Y_train):

    XX_train, YY_train = X_train[train_index], Y_train[train_index]
    XX_cv, YY_cv = X_train[cv_index], Y_train[cv_index]
    
    rf.fit(XX_train, YY_train)
    ypred = rf.predict_proba(X_test)
    result += ypred[:,1]

    train_pred = rf.predict_proba(XX_cv)
    train_result[cv_index] = train_pred[:,1]
    print(roc_auc_score(YY_cv, train_pred[:,1]))

result /= 5


submit = pd.DataFrame({'click_id':click_id, 'is_attributed':result})
submit.to_csv('submit.csv', index=False)
train_result = pd.DataFrame({"RandomForest":train_result})
train_result.to_csv("train_predict.csv", index=False)


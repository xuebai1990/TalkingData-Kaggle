import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

Y_day9 = pd.read_csv("../feature/train-day9-total.csv", usecols=["is_attributed"])
Y_pred = np.loadtxt("fm_predict.dat")

print(roc_auc_score(Y_day9, Y_pred[:,1]))

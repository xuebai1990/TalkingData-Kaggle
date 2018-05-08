import pandas as pd
import numpy as np
import gc
from dtypes import dtypes

dict = {1:'train-day6',2:'train-day7',3:'train-day8',4:'train-day9',5:'test'}

def combine(count):
    data_total = pd.read_csv("../feature/"+dict[count]+"-total.csv", dtype=dtypes)
    data_uniq = pd.read_csv("../feature/"+dict[count]+"-uniq.csv", dtype=dtypes)
    data = pd.concat([data_total, data_uniq], axis=1)
    data.to_csv("../feature/"+dict[count]+"-total.csv", index=False)
    del data_total, data_uniq, data
    gc.collect()

for i in range(1, 6):
    combine(i)
    print("Finished!")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

print(pd.isnull(train).sum())
print(pd.isnull(test).sum())


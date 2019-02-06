
# for data analysis
import pandas as pd
import numpy as np


# for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

#import the train and test files (CSV)

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

print(train.describe(include="all"))

#11 rows x 12 columns, a lot of NaN values





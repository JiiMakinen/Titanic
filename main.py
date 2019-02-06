
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

# print(train.describe(include="all"))

# 11 rows x 12 columns, a lot of NaN values

# How does the data look like

#print(train.columns)
# -->
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')

# print(train.sample(5))
# print(train.dtypes)
#-->
# PassengerId      int64
# Survived         int64
# Pclass           int64
# Name            object
# Sex             object
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object
# Fare           float64
# Cabin           object
# Embarked        object
print(pd.isnull(train).sum())

# Observations from the dataset:
# 891 passengers
# Age feature is missing 177 values (19.86 %), we'll need to fill these Nan values.
# Embarked feature is missing 2 values (0.22%), we can dismiss as relatively harmless error.

# Cabin feature is missing 687 values (77.1 %), too much information is missing, cant really fill the Nan values.
# Conclusion is that Age feature could be very important to survival.
# Cabin features are excluded for now in the project as there's no way to accurately fill the gaps.

# Predictions:
# Females and really young children have a higher chance of survival as they were the first ones in the lifeboats.
# Males have a low survival rate
# People with relatives on board have lower survival rate as they would look after each other (Sibsp/Parch)
# People in higher class cabins have better survival rate as they were taken care of better by the crew. (Pclass)












import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# print(pd.isnull(train).sum())
# print(pd.isnull(test).sum())

# Missing some values that need to be filled in.
# There are also some useless information like cabin, ticket number and name.

# Lets drop the ticket and cabin numbers.

train = train.drop(columns=['Ticket', 'Cabin'])
test = test.drop(columns=['Ticket', 'Cabin'])

# print(train.columns)

# The training set is missing two Embarked values. they can be easily given the most dominant value of the three.

print("Embarked from Southampton: ", train[train["Embarked"] == "S"].shape[0])
print("Embarked from Cherbourg: ", train[train["Embarked"] == "C"].shape[0])
print("Embarked from Queenstown: ", train[train["Embarked"] == "Q"].shape[0])

# Most of the people aboard embarked from Southhampton. Thus we can safely fill the missing two embarkment values
# to be "S"

train = train.fillna({"Embarked": "S"})

# print(pd.isnull(train).sum())
# print(pd.isnull(test).sum())

# Now we still have a lot of missing Age values. I'm going to fill these values with the average of the Pclass. As in,
# calculate the average age of every passenger class. Then fill the ageless first class passengers with the average
# of 1st class passengers and so on. We'll claculate the averages from the training data.

class1 = train[train["Pclass"] == 1]
age1mean = class1["Age"].sum()/len(class1)
print(age1mean)

class2 = train[train["Pclass"] == 2]
age2mean = class2["Age"].sum()/len(class2)
print(age2mean)

class3 = train[train["Pclass"] == 3]
age3mean = class3["Age"].sum()/len(class3)
print(age3mean)


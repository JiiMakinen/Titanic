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
# This may not be the optimal solution for filling the ages, could be improved.


age1mean = train[train["Pclass"] == 1]["Age"].mean()
print(age1mean)

age2mean = train[train["Pclass"] == 2]["Age"].mean()
print(age2mean)

age3mean = train[train["Pclass"] == 3]["Age"].mean()
print(age3mean)

# Fill missing values in training data
for i in range(len(train["PassengerId"])):
    if pd.isnull(train["Age"][i]):
        if train["Pclass"][i] == 1:
            train["Age"][i] = age1mean
        elif train["Pclass"][i] == 2:
            train["Age"][i] = age2mean
        elif train["Pclass"][i] == 3:
            train["Age"][i] = age3mean



# Fill values in the test data.

for i in range(len(test["PassengerId"])):
    if pd.isnull(test["Age"][i]):
        if test["Pclass"][i] == 1:
            test["Age"][i] = age1mean
        elif test["Pclass"][i] == 2:
            test["Age"][i] = age2mean
        elif test["Pclass"][i] == 3:
            test["Age"][i] = age3mean



# Lets fill the missing fare value from test data with the mean of its class

dfnan = test[test.isnull().any(axis=1)]
print(dfnan)

# The missing Fare value is from class 3, passangerid 1044 and row 152, lets fill it.

test["Fare"][152] = train[train["Pclass"] == 3]["Fare"].mean()

print(pd.isnull(train).sum())
print(pd.isnull(test).sum())

# Missing data has been filled, now assing numerical groups to sex, fare, embarkment







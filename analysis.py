import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


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

# print("Embarked from Southampton: ", train[train["Embarked"] == "S"].shape[0])
# print("Embarked from Cherbourg: ", train[train["Embarked"] == "C"].shape[0])
# print("Embarked from Queenstown: ", train[train["Embarked"] == "Q"].shape[0])

# Most of the people aboard embarked from Southhampton. Thus we can safely fill the missing two embarkment values
# to be "S"

train = train.fillna({"Embarked": "S"})

# print(pd.isnull(train).sum())
# print(pd.isnull(test).sum())

# Now we still have a lot of missing Age values. I'm going to fill these values with the average of the Pclass. As in,
# calculate the average age of every passenger class. Then fill the ageless first class passengers with the average
# of 1st class passengers and so on. We'll calculate the averages from the training data.
# This may not be the optimal solution for filling the ages, could be improved.


age1mean = train[train["Pclass"] == 1]["Age"].mean()
# print(age1mean)

age2mean = train[train["Pclass"] == 2]["Age"].mean()
# print(age2mean)

age3mean = train[train["Pclass"] == 3]["Age"].mean()
# print(age3mean)

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
# print(dfnan)

# The missing Fare value is from class 3, passangerid 1044 and row 152, lets fill it.

test["Fare"][152] = train[train["Pclass"] == 3]["Fare"].mean()

# print(pd.isnull(train).sum())
# print(pd.isnull(test).sum())

# Missing data has been filled, now assing numerical groups to sex, fare, embarkment

sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

train['Sex'] = train['Sex'].map(sex_map)
test['Sex'] = test['Sex'].map(sex_map)

# Make categories for the fare and map them

train['FareGroup'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareGroup'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])

# Now we can drop the name and fare values from the data as they are not needed.

train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)

# print(train.head(3))
# print(test.head(3))

# Now we cant start analyzing the data.

# DATA ANALYSIS:

# I will be using 50% of the training data to test the accuracy of different machine learning models.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

predict = train.drop(columns=['Survived', 'PassengerId'])
target = train['Survived']

x_train, x_test, y_train, y_test = train_test_split(predict, target, test_size=0.5, random_state=0)

# I will be using the following models and testing their accuracy on the training data:
# Gaussian Naive Bayes
# Decision Tree Classifier
# Random Forest Classifier
# k-Nearest Neighbors
# Gradient Boosting Classifier

# Gaussian Naive Bayes:

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()  # initialze gaussian
gaussian.fit(x_train, y_train)   # Fit training data
y_predict = gaussian.predict(x_test)  #predict y from x_test
gaussian_accuracy = round(accuracy_score(y_predict, y_test) * 100, 2)  #get accuracy of the prediction w.r.t y_test.
# print(gaussian_accuracy)

# Decision Tree Classifier:
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
y_predict = dtree.predict(x_test)
decisionTree_accuracy = round(accuracy_score(y_predict, y_test) * 100, 2)
# print(decisionTree_accuracy)

# Random Forest Classifier:

from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier()
rforest.fit(x_train, y_train)
y_predict = rforest.predict(x_test)
randomForest_accuracy = round(accuracy_score(y_predict, y_test) * 100, 2)
# print(randomForest_accuracy)

# k-nearest Neighbours:

from sklearn.neighbors import KNeighborsClassifier

kNeigh = KNeighborsClassifier()
kNeigh.fit(x_train, y_train)
y_predict = kNeigh.predict(x_test)
kNeighbours_accuracy = round(accuracy_score(y_predict, y_test) * 100, 2)
# print(kNeighbours_accuracy)

# Gradient Boosting Classifier:

from sklearn.ensemble import GradientBoostingClassifier

gBoost = GradientBoostingClassifier()
gBoost.fit(x_train, y_train)
y_predict = gBoost.predict(x_test)
gBoost_accuracy = round(accuracy_score(y_predict, y_test) * 100, 2)
# print(gBoost_accuracy)


# Visualising the accuracy rates of the models:

df1 = pd.DataFrame({
    'Model': ['KNN', 'Random Forest', 'Naive Bayes', 'Decision Tree', 'G-Boost Classifier'],
    'Accuracy': [kNeighbours_accuracy, randomForest_accuracy, gaussian_accuracy, decisionTree_accuracy, gBoost_accuracy]})

y = df1["Accuracy"]
xlocs = [0, 1, 2, 3, 4]
plt.figure(figsize=(8, 7))
models_plot = sns.barplot(x="Model", y="Accuracy", data=df1, palette="muted")

for i in range(5):
    plt.text(xlocs[i]-0.3, y[i]+2, str(y[i]))

models_plot.set_xticklabels(models_plot.get_xticklabels(), rotation=15, ha="right")
plt.show()

# Here we can see that the best model for predicting the survivors is the Gradient Boosting Classifier.
# Lets use it to predict the survival of the whole test group in test.csv.


test_id = test['PassengerId']

prediction = gBoost.predict(test.drop(columns='PassengerId'))

results = pd.DataFrame({'PassengerId': test_id, 'Survived': prediction})


print(round(results['Survived'].mean()*100, 2), "% of the passengers in the test file could have survived.")

# We'll conclude that almost 40% of the test file passengers could have survived the Titanic disaster. This estimate is
# rough as the training data was missing a lot of values which had to be filled.

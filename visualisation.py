
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
#print(pd.isnull(train).sum())

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
# People with relatives on board have lower survival rate as they would look after each other (SibSp/Parch)
# People in higher socioeconomic classes (Cabin classes) have better survival rate as they
# were taken care of better by the crew.


# Visualisations of the predictions:

# Men vs. Female in different classes:
sns.set(style="whitegrid")
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Survival probability")
plt.show()

# This shows that females had better survival rate in every passenger class.
# The female passengers of 1st class had almost 100% survival chance.
# The plot also shows that the survival rate decreased as the class got worse. As in, the 1st class passengers
# had a higher chance of survival compared to the 2nd and 3rd class passengers.

# Comparison of the survival chance of people with siblings or spouses (SibSp):

f = sns.catplot(x="SibSp", y="Survived", data=train,kind="bar", palette="muted")
f.set_ylabels("Survival probability")
plt.show()

# This plot shows that people with a lot of siblings or spouses aboard had less chance of survival.
# Interestingly, the people with no siblings or spouses had lower chance than the people with 1 or 2.
# This could be explained by peoples ability to work better in small groups. If you had one or two siblings or spouses,
# you would be communicating with each other better and get quicker to the lifeboats than people with neither.
# If the amount of siblings or spouses grew too big, the chance of survival drops as it would be hard to communicate and
# get everyone to safety as well as with only 1 or 2 sibling or spouses.

d = sns.catplot(x="Parch", y="Survived", data=train, kind="bar", palette="muted")
d.set_ylabels("Survival probability")
plt.show()

# Here we can see that if the number of parents/Children were 1,2 or 3, the survival chance would be higher than else's.
# This could be due to empathy towards parents. If parents had one or two kids, they would get taken care by others and
# escorted quickly to the lifeboats. People who were alone or children with no parents had a hard time surviving.
# Without anyone to help you, you have less chance to survive. Children without parents had a hard time realising what
# was going on. Again, if the number of parents / children grew too big, it would be hard to communicate and get
# everyone to safety.

plt.figure(figsize=(8,7))
train["Age"] = train["Age"].fillna(-0.5)
breaks = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], breaks, labels=labels)
s = sns.barplot(x="AgeGroup", y="Survived", data=train)
s.set_xticklabels(s.get_xticklabels(), rotation=40, ha="right")
plt.show()

# This shows that babies were most likely to survive as they would probably be carried by their mothers
# who also had a high chance of survival. Seniors had a lower chance of survival than everyone else as they probably
# couldn't survive the cold waters and exhaustion. Other agegroups had almost the same chance of survival.









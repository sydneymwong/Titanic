import scipy
import numpy
import pandas
from sklearn import tree

train = pandas.read_csv('train.csv')
list(train)
train = train.fillna(0)

train['Sex'] = train['Sex'].replace("male", "0")
train['Sex'] = train['Sex'].replace("female", "1")
train['Sex'] = train['Sex'].apply(int)

train_features = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

#train_features = train.drop(['Survived'], axis=1)
#train_features_dummies = pandas.get_dummies(train_features)
#survived = train['Survived']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, train['Survived'])
#clf = clf.fit(train_features_dummies, train['Survived'])
train_predictions = clf.predict(train_features)
numpy.sum(train_predictions == train['Survived'])

# Read in the test data
test = pandas.read_csv('test.csv')
test = test.fillna(0)

test['Sex'] = test['Sex'].replace("male", "0")
test['Sex'] = test['Sex'].replace("female", "1")
test['Sex'] = test['Sex'].apply(int)

test_features = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

#test_features_dummies = pandas.get_dummies(test)
test_predictions = clf.predict(test_features)




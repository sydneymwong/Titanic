import scipy
import numpy
import pandas
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# read in training data
train = pandas.read_csv('train.csv')
list(train)

# rudimentary cleaning
train = train.fillna(0)

train['Sex'] = train['Sex'].apply(lambda value: 1 if value == "female" else "0")

# for now, limit training set to numeric variables

train_features = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# create a classification tree using the training data

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, train['Survived'])
#clf = clf.fit(train_features_dummies, train['Survived'])
train_predictions = clf.predict(train_features)
numpy.sum(train_predictions == train['Survived'])

# read in the test data
test = pandas.read_csv('test.csv')

# repeat same cleaning/subsetting on the test data
test = test.fillna(0)

test['Sex'] = test['Sex'].apply(lambda value: 1 if value == "female" else "0")

test_features = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

test_predictions = clf.predict(test_features)
predictions = pandas.DataFrame(test['PassengerId'])
predictions = predictions.assign(Survived = test_predictions)

# export pandas dataframe as csv
predictions.to_csv('submission1.csv', index = False)

# random forest
clfrf = RandomForestClassifier(n_jobs=1)
clfrf.fit(train_features, train['Survived'])
train_predictions_rf = clfrf.predict(train_features)
numpy.sum(train_predictions_rf == train['Survived'])
test_predictions_rf = clfrf.predict(test_features)
predictions_rf = pandas.DataFrame(test['PassengerId'])
predictions_rf = predictions_rf.assign(Survived = test_predictions_rf)

#export pandas dataframe as csv
predictions_rf.to_csv('submission2.csv', index = False)

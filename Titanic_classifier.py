import scipy
import numpy
import pandas
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_train():
    # read in training data
    train = pandas.read_csv('train.csv')
    list(train)
    # rudimentary cleaning
    train = train.fillna(0)
    train['Sex'] = train['Sex'].apply(lambda value: 1 if value == "female" else "0")
    # for now, limit training set to numeric variables
    train_features = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    return train_features, train['Survived']


def load_test():
    # read in the test data
    test = pandas.read_csv('test.csv')
    # repeat same cleaning/subsetting on the test data
    test = test.fillna(0)
    test['Sex'] = test['Sex'].apply(lambda value: 1 if value == "female" else "0")
    test_features = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    return test_features, test['PassengerId']

def return_classifier(name):
    if name == "tree":
        clf = tree.DecisionTreeClassifier()
    elif name == "rf":
        clf = RandomForestClassifier(n_jobs=1)
    elif name == "log":
        clf = LogisticRegression()
    return clf

def apply_classifier(model, train_feat, train_labels, test_feat, test_names, submission_number):
    clf = model.fit(train_feat, train_labels)
    train_predictions = clf.predict(train_feat)
    numpy.sum(train_predictions == train_labels)

    test_predictions = clf.predict(test_feat)
    predictions = pandas.DataFrame(test_names)
    predictions = predictions.assign(Survived=test_predictions)

    # export pandas dataframe as csv
    predictions.to_csv('submission' + submission_number + '.csv', index=False)

def main():
    train_features, train_labels = load_train()
    test_features, test_names = load_test()
    #single tree classifier
    clftree = return_classifier("tree")
    apply_classifier(clftree, train_features, train_labels, test_features, test_names, "1")
    #random forest classifier
    clfrf= return_classifier("rf")
    apply_classifier(clfrf, train_features, train_labels, test_features, test_names, "2")
    #logistic regression classifier
    clflog = return_classifier("log")
    apply_classifier(clflog, train_features, train_labels, test_features, test_names, "3")

if __name__ == "__main__" :
    main()

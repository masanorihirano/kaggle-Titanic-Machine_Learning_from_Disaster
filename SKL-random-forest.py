#!-- coding:utf-8 --
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

train["Name_len"]=train["Name"].apply(lambda x: len(x))
test["Name_len"]=test["Name"].apply(lambda x: len(x))

def correct_data(train_data, test_data):
    
    # Make missing values ​​for training data from test data as well
    train_data.Age = train_data.Age.fillna(test_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(test_data.Fare.median())
    train_data.Name_len = train_data.Name_len.fillna(test_data.Name_len.median())

    test_data.Age = test_data.Age.fillna(test_data.Age.median())
    test_data.Fare = test_data.Fare.fillna(test_data.Fare.median())    
    train_data = correct_data_common(train_data)
    test_data = correct_data_common(test_data)    
    return train_data,  test_data

def correct_data_common(titanic_data):
    titanic_data.Sex = titanic_data.Sex.replace(['male', 'female'], [0, 1])
    titanic_data.Embarked = titanic_data.Embarked.fillna("S")
    titanic_data.Embarked = titanic_data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

    return titanic_data

train_data,  test_data = correct_data(train, test)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name_len"]

parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'random_state'      : [0],
}
gsc = GridSearchCV(RandomForestClassifier(), parameters,cv=3)
gsc.fit(train_data[predictors], train_data["Survived"])

alg = RandomForestClassifier(verbose= True,max_depth=gsc.best_params_['max_depth'],n_estimators=gsc.best_params_['n_estimators'],random_state=gsc.best_params_['random_state'],n_jobs=-1)
alg.fit(train_data[predictors], train_data["Survived"])

predictions = alg.predict(test_data[predictors])

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('submission.csv', index=False)


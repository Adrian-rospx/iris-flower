# Machine learning exercise
# graphing the iris flower dataset
# and uncovering the patterns with machine learning algorithms

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from display_utils import (display_all, display_length, display_width)

# preparing the data
iris = load_iris()
iris_data = pd.DataFrame(data = iris.data, columns = iris.feature_names)
# target data
iris_data["target"] = iris.target

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(iris_data.loc[:, iris_data.columns != "target"],
                                                    iris_data["target"],
                                                    test_size = 0.2)

# initialise logistic regression model
model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)
# realise prediction
prediction = model.predict(X_test)

# turn data to pandas info
y_test = pd.Series(y_test, 
                   name = "target")
prediction = pd.Series(prediction, 
                       index = y_test.index, 
                       name = "target")
# create output data
test_data = pd.concat([X_test, y_test], axis = 1)
prediction_data = pd.concat([X_test, prediction], axis = 1)

# display test data and prediction data
display_length(test_data)
display_length(prediction_data)
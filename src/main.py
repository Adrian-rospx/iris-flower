# Machine learning exercise
# graphing the iris flower dataset
# and uncovering the patterns with machine learning algorithms

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data_prep import get_iris_data
from display_utils import display_all, display_length, display_width
from evaluate import evaluate_results

iris_data, iris_target = get_iris_data()

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_target,
                                                    test_size = 0.2)

# initialise logistic regression model
model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)
# realise prediction
prediction = model.predict(X_test)

# turn data to pandas series
y_test = pd.Series(y_test, 
                   name = "target")
prediction = pd.Series(prediction, 
                       index = y_test.index, 
                       name = "target")

# display test data and prediction data
display_all(X_test, y_test)
display_all(X_test, prediction)

evaluate_results(y_test, prediction)
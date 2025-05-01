""" 
    Machine learning exercise

    graphing the iris flower dataset
    and displaying the patterns 
    by using machine learning algorithms

    supervised learning: 
        linear regression
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
# user defined:
from data_prep import get_iris_data
from display_utils import display_all, display_reg, display_pca
from ml_models import logistic_regression
from evaluate import evaluate_results

iris_data, iris_target = get_iris_data()

# split train and test data
X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                    iris_target,
                                                    test_size = 0.2)
# turn output data to pandas series
y_test = pd.Series(y_test, 
                name = "target")

def main():
    # display pca plot of all data
    display_pca(iris_data, iris_target)

    # logistic regression
    y_prediction = logistic_regression(X_train, X_test, y_train, y_test)

    # display normal plot of training data
    display_reg(X_train, y_train)

    # display evaluation stats
    evaluate_results(y_test, y_prediction)

    # display test data and prediction data
    display_all(X_test, y_test)
    display_all(X_test, y_prediction)


if __name__ == "__main__":
    main()
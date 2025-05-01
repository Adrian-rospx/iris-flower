""" 
    Machine learning exercise

    graphing the iris flower dataset
    and displaying the patterns 
    by using machine learning algorithms
"""

import pandas as pd
from sklearn.model_selection import train_test_split
# user defined:
from data_prep import get_iris_data
from display_utils import display_all, display_reg, display_pca
from ml_models import logistic_regression, k_nearest_neighbors
from evaluate import evaluate_results

def main():
    """Basic data preparation"""
    # get iris flower data
    iris_data, iris_target = get_iris_data()

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(iris_data,
                                                        iris_target,
                                                        test_size = 0.2)
    # turn output data to pandas series
    y_test = pd.Series(y_test, name = "target")

    # display pca plot of all data
    display_pca(iris_data, iris_target)

    """ Logistic regression visualisation
        K-Nearest Neighbors visualisation
    """
    # logistic regression
    y_lr_pred = logistic_regression(X_train, X_test, y_train, y_test, c=1)
    # k nearest neighbors
    y_knn_pred = k_nearest_neighbors(X_train, X_test, y_train, y_test, k=5)

    # display normal plot of training data
    display_reg(X_train, y_train)

    # display evaluation stats
    print("Logistic regression:")
    evaluate_results(y_test, y_lr_pred)
    print('\nK-Nearest Neighbors:')
    evaluate_results(y_test, y_knn_pred)

    # display test data and prediction data
    display_all(X_test, y_test)
    display_all(X_test, y_lr_pred)
    display_all(X_test, y_knn_pred)


if __name__ == "__main__":
    main()
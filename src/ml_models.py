import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def pca(iris_data) -> pd.DataFrame:
    """PCA dimensionality reduction to a 2d matrix
    as a Pandas DataFrame
    """
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(iris_data)
    coordinates = pd.DataFrame(coordinates)
    
    return coordinates

def logistic_regression(X_train, X_test, 
                        y_train, y_test, 
                        c: float = 1.0) -> pd.Series:
    """return prediction of the 
    logistic regression machine learning model"""
    # initialise logistic regression model
    model = LogisticRegression(max_iter = 200, C=c)
    model.fit(X_train, y_train)
    # realise prediction
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, 
                        index = y_test.index, 
                        name = "target")
    return y_pred

def k_nearest_neighbors(X_train, X_test, 
                        y_train, y_test,
                        k: int = 5) -> pd.Series:
    """return prediction of k nearest neighbors model"""
    # initialise and train
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X = X_train, y=y_train)
    # realise prediction
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred, 
                        index = y_test.index, 
                        name = "target")
    return y_pred
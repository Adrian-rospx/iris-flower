"""functions for preparing model data"""

import pandas as pd
from sklearn.datasets import load_iris

def get_iris_data() -> tuple[pd.DataFrame, pd.Series]:
    """return iris data in the form of a tuple\n
    iris_data, iris.target = get_iris_data()"""
    iris = load_iris()
    iris_data = pd.DataFrame(data = iris.data, columns = iris.feature_names)
    # target data
    iris_target = pd.Series(iris.target, 
                            index = iris_data.index,
                            name = "target")
    return iris_data, iris_target
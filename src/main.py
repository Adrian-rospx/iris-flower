# Machine learning exercise
# graphing the iris flower dataset
# and uncovering the patterns with machine learning algorithms

import pandas as pd
from sklearn.datasets import load_iris

from display_utils import (display_all, display_length, display_width)

# preparing the data
iris = load_iris()
iris_data = pd.DataFrame(data = iris.data, columns = iris.feature_names)
# target data
iris_data["target"] = iris.target

display_all(iris_data)
display_length(iris_data)
display_width(iris_data)
# Machine learning exercise
# graphing the iris flower dataset
# and uncovering the patterns with machine learning algorithms

import pandas as pd
from sklearn.datasets import load_iris

# preparing the data
iris = load_iris()
iris_data = pd.DataFrame(data = iris.data, columns = iris.feature_names)
# add target labels
iris_data['target'] = iris.target

print(iris_data)
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
# change target numbers to flower names
iris_data['target_name'] = iris_data['target'].apply(lambda x: iris.target_names[x])

print(iris_data)
print(iris_data.head())
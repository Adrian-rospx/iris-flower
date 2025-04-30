# Machine learning exercise
# graphing the iris flower dataset
# and uncovering the patterns with machine learning algorithms

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# preparing the data
iris = load_iris()
iris_data = pd.DataFrame(data = iris.data, columns = iris.feature_names)
# add target labels
iris_data['target'] = iris.target
# change target numbers to flower names
iris_data['target_name'] = iris_data['target'].apply(lambda x: iris.target_names[x])

# use seaborn for display and visualisation
sns.set_theme()

sns.pairplot(iris_data, hue = "target_name")
plt.show()

iris_lengths = iris_data.loc[:, ["sepal length (cm)", "petal length (cm)"]]
sns.pairplot(iris_lengths)
plt.show()

iris_widths = iris_data.loc[:, ["sepal width (cm)", "petal width (cm)"]]
sns.pairplot(iris_widths)
plt.show()
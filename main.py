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
# target data
iris_data["target"] = iris.target

# use seaborn for display and visualisation
sns.set_theme()

sns.pairplot(iris_data, hue="target")
plt.show()

# load and display length data
iris_lengths = iris_data.loc[:, ["sepal length (cm)", "petal length (cm)", 'target']]
sns.relplot(
    data = iris_lengths,
    x = "sepal length (cm)",
    y = "petal length (cm)",
    hue = "target"
)
plt.show()

# display width data
iris_widths = iris_data.loc[:, ["sepal width (cm)", "petal width (cm)", "target"]]
sns.relplot(
    data = iris_widths,
    x = "sepal width (cm)",
    y = "petal width (cm)",
    hue = "target"
)
plt.show()
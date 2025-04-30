"""User defined utilities for displaying graphs"""
import matplotlib.pyplot as plt
import seaborn as sns

# use seaborn for display and visualisation
sns.set_theme()

def display_all(iris_data):
    sns.pairplot(iris_data, hue="target")
    plt.show()

def display_length(iris_data):
    """display length data"""
    sns.relplot(
        data = iris_data,
        x = "sepal length (cm)",
        y = "petal length (cm)",
        hue = "target"
    )
    plt.show()

def display_width(iris_data):
    """display width data"""
    sns.relplot(
        data = iris_data,
        x = "sepal width (cm)",
        y = "petal width (cm)",
        hue = "target"
    )
    plt.show()
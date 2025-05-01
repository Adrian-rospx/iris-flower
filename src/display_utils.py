"""User defined utilities for displaying graphs"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ml_models import pca

# use seaborn for display and visualisation
sns.set_theme()

def display_pca(data, target_data) -> None:
    """display PCA scatter plot of data"""
    coords = pca(data)
    coords = pd.DataFrame(coords)
    coords = pd.concat([coords, target_data], axis=1)

    sns.scatterplot(coords, x=0, y=1, hue="target")
    plt.show()

def display_all(iris_data, target_data: pd.Series = pd.Series()):
    """display all data in a pairplot"""
    if target_data.empty:
        sns.pairplot(iris_data)
    else:
        data = pd.concat([iris_data, target_data], axis = 1)
        sns.pairplot(data, hue="target", corner = True)
    plt.show()

def display_reg(iris_data, target_data: pd.Series = pd.Series()):
    """display all data in a pairplot as a linear regression"""
    if target_data.empty:
        sns.pairplot(iris_data)
    else:
        data = pd.concat([iris_data, target_data], axis = 1)
        sns.pairplot(data, hue="target", corner = True, kind="reg")
    plt.show()

def display_length(iris_data, target_data: pd.Series | None = None):
    """display length data"""
    if target_data.empty:
        sns.pairplot(
            iris_data,
            x = "sepal length (cm)",
            y = "petal length (cm)"
        )
    else:
        data = pd.concat([iris_data, target_data], axis = 1)
        sns.pairplot(
            data,
            x = "sepal length (cm)",
            y = "petal length (cm)",
            hue = "target"
        )
    plt.show()

def display_width(iris_data, target_data: pd.Series | None = None):
    """display width data"""
    if target_data.empty:
        sns.pairplot(
            iris_data,
            x = "sepal width (cm)",
            y = "petal width (cm)"
        )
    else:
        data = pd.concat([iris_data, target_data], axis = 1)
        sns.pairplot(
            data,
            x = "sepal width (cm)",
            y = "petal width (cm)",
            hue = "target"
        )
    plt.show()

def display_confusion_matrix(cm: list) -> None:
    """display confusion matrix"""
    sns.heatmap(cm)
    plt.show()
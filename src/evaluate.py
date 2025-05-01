import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from display_utils import display_confusion_matrix

def evaluate_results(y_test: pd.DataFrame, y_pred: pd.DataFrame):
    """display accuracy score and confusion matrix data"""
    accuracy_scr = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:\n{accuracy_scr * 100:.2f}%\n")
    print(f"Confusion matrix:\n{confusion_mat}")

    display_confusion_matrix(confusion_mat)
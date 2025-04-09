import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve


def plot_confusion(y_true, y_pred, perc: float = None) -> None:
    """
    Plots a confusion matrix for binary classification.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities or binary labels.
        perc (float, optional): Threshold to convert probabilities to class labels.
                                If None, y_pred is treated as already classified.
    """
    if perc is not None:
        y_class = [1 if p >= perc else 0 for p in y_pred]
        print(classification_report(y_true, y_class))
        print(f"Threshold Used: {perc}")
        cm = confusion_matrix(y_true, y_class)
    else:
        cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Churn", "Churn"],
                yticklabels=["Not Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc(name: str, labels, predictions, **kwargs) -> None:
    """
    Plots a ROC curve.

    Args:
        name (str): Label for the ROC curve.
        labels (array-like): True binary labels.
        predictions (array-like): Predicted probabilities.
        **kwargs: Additional keyword arguments for matplotlib `plot`.
    """
    fpr, tpr, _ = roc_curve(labels, predictions)
    plt.plot(100 * fpr, 100 * tpr, label=name, linewidth=2, **kwargs)
    plt.xlabel("False Positives [%]")
    plt.ylabel("True Positives [%]")
    plt.title("ROC Curve")
    plt.grid(True)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

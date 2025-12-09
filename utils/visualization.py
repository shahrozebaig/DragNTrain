import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_confusion_matrix_figure(cm, class_names):
    max_classes_to_show = 15

    if len(class_names) > max_classes_to_show:
        cm = cm[:max_classes_to_show, :max_classes_to_show]
        class_names = class_names[:max_classes_to_show]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix (Top Classes)")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0 if cm.max() != 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    return fig

def plot_accuracy_bar_figure(accuracy):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["Accuracy"], [accuracy * 100])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy")
    ax.text(0, accuracy * 100 + 1, f"{accuracy * 100:.1f}%", ha="center")
    fig.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["Feature"], df["Importance"])
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    fig.tight_layout()

    return fig, df

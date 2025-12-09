import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

def train_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series):
    if X_train.shape[1] == 0:
        raise ValueError("No valid feature columns available after preprocessing.")
    if y_train.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=3000, solver="lbfgs")
    elif model_type == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    if X_test.shape[1] == 0:
        raise ValueError("No valid feature columns available for testing.")
    if y_test.nunique() < 2:
        raise ValueError("Target column must contain at least two classes.")
    imputer = SimpleImputer(strategy="mean")
    X_test = imputer.fit_transform(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    class_names = [str(c) for c in sorted(y_test.unique())]
    return acc, report, cm, class_names

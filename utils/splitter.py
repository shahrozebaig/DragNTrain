import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.8,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
    )
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    y_train = y_train.fillna(y_train.mode()[0])
    y_test = y_test.fillna(y_test.mode()[0])

    return X_train, X_test, y_train, y_test

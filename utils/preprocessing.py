import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def preprocess_features(
    df: pd.DataFrame,
    target_col: str,
    method: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    processed_df = df.copy()
    processed_df = processed_df.loc[:, processed_df.notna().any()]
    feature_cols = [c for c in processed_df.columns if c != target_col]
    numeric_cols = processed_df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = processed_df[feature_cols].select_dtypes(exclude="number").columns.tolist()
    scaled_cols = []
    for col in numeric_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    for col in categorical_cols:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    processed_df = processed_df.dropna(subset=[target_col])
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))
    if method is not None and numeric_cols:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unknown preprocessing method")
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
        scaled_cols = numeric_cols
    return processed_df, scaled_cols

import pandas as pd

def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine="python",
                    encoding="utf-8",
                    skip_blank_lines=True
                )
            except:
                df = pd.read_csv(
                    uploaded_file,
                    sep=None,
                    engine="python",
                    encoding="latin1",
                    skip_blank_lines=True
                )

        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
        if df.empty or df.shape[1] == 0:
            raise ValueError("Uploaded file contains no readable columns.")
    except Exception as e:
        raise ValueError(f"File could not be read: {e}")
    return df

def get_basic_info(df: pd.DataFrame):
    rows, cols = df.shape
    col_names = list(df.columns)
    return rows, cols, col_names

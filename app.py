import streamlit as st
import pandas as pd
from utils.data_loader import load_data, get_basic_info
from utils.preprocessing import preprocess_features
from utils.splitter import split_data
from utils.model import train_model, evaluate_model
from utils.visualization import (
    plot_confusion_matrix_figure,
    plot_accuracy_bar_figure,
    plot_feature_importance,
)

st.set_page_config(
    page_title="DragNTrain",
    page_icon="🧠",
    layout="wide",
)


def load_custom_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_custom_css()


def reset_pipeline():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


if "df" not in st.session_state:
    st.session_state.df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "split_data" not in st.session_state:
    st.session_state.split_data = None
if "model_info" not in st.session_state:
    st.session_state.model_info = None
if "model_history" not in st.session_state:
    st.session_state.model_history = []
if "step" not in st.session_state:
    st.session_state.step = 0


with st.sidebar:
    try:
        st.image("assets/logo.png", width=120)
    except:
        st.markdown("DragNTrain")

    st.markdown("<h2 class='sidebar-title'>DragNTrain</h2>", unsafe_allow_html=True)
    st.markdown("### Your No-Code Machine Learning Studio")
    st.caption("Build, train & evaluate ML models visually — zero coding required.")

    step_labels = {
        1: "📂 Upload Dataset",
        2: "🧹 Preprocess Data",
        3: "✂️ Train–Test Split",
        4: "🤖 Train & Evaluate Model",
    }

    if st.session_state.step > 0:
        st.markdown("### 🔁 Workflow Progress")
        for i in range(1, 5):
            if st.session_state.step > i:
                css_class = "step-completed"
            elif st.session_state.step == i:
                css_class = "step-active"
            else:
                css_class = "step-pending"

            st.markdown(
                f"<div class='{css_class}'>{step_labels[i]}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        if st.button("🔄 Reset Entire Pipeline"):
            reset_pipeline()

    st.markdown("---")
    st.markdown(
        "<p class='sidebar-footer'>© 2025 • DragNTrain AI • Built for Learning & Rapid ML Prototyping</p>",
        unsafe_allow_html=True,
    )


if st.session_state.step == 0:
    st.markdown(
        """
        <div class='header'>
            <div>
                <h1>🧠 DragNTrain</h1>
                <h3>No-Code Machine Learning Platform</h3>
                <p>
                Build, train, evaluate and compare machine learning models using a fully visual workflow.
                No coding. No setup. Just upload your dataset and start experimenting.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("## 🚀 Why DragNTrain Exists")
    st.write("""
    Machine learning is powerful, but traditional workflows are complex and intimidating.
    DragNTrain removes this barrier by transforming the entire ML lifecycle into a simple,
    guided, and visual experience. Whether you're a student, beginner, or working professional,
    you can now build real ML models confidently.
    """)

    st.markdown("---")

    st.markdown("## 🛠 What You Can Do")
    st.write("""
    • Upload real-world datasets (CSV / Excel)  
    • Automatically clean and preprocess features  
    • Select the target variable visually  
    • Split data into training and testing sets  
    • Train ML models in one click  
    • View accuracy, confusion matrix & feature importance  
    • Compare multiple trained models  
    • Download prediction results instantly  
    """)

    st.markdown("---")

    st.markdown("## 🎯 Who Should Use This?")
    st.write("""
    ✅ Students building academic projects  
    ✅ Beginners learning machine learning concepts  
    ✅ Data analysts validating ideas quickly  
    ✅ Developers testing datasets visually  
    """)

    st.markdown("---")

    st.markdown("## ⚙️ How DragNTrain Works")
    st.write("""
    The workflow is divided into four simple steps:
    
    1️⃣ Upload your dataset  
    2️⃣ Preprocess & select the target  
    3️⃣ Perform train–test split  
    4️⃣ Train and evaluate your ML model  

    Each step is clearly tracked so you always know where you are in the pipeline.
    """)

    st.markdown("---")

    st.markdown("## 🔐 Privacy & Security")
    st.write("""
    Your data is processed only during your active session.
    No permanent storage, no tracking, and no external data usage.
    """)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Visual ML", "100%")

    with c2:
        st.metric("Auto Processing", "Enabled")

    with c3:
        st.metric("One-Click Training", "Live")

    st.markdown("---")

    if st.button("🚀 Start Building Your ML Model"):
        st.session_state.step = 1
        st.rerun()

    st.stop()


st.markdown(
    """
    <div class='header'>
        <div>
            <h1>🧠 DragNTrain</h1>
            <p><b>Your Visual Machine Learning Playground</b></p>
            <p>Create an end-to-end ML workflow with just clicks — no coding required.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

progress_map = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
st.progress(progress_map.get(st.session_state.step, 0.25))

st.markdown("---")


def step_upload_data():
    st.subheader("Step 1 · Upload Your Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx"],
            help="This will be your dataset for the ML pipeline.",
        )

        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                st.session_state.df = df
                st.success("File uploaded and loaded successfully")

                rows, cols, col_names = get_basic_info(df)

                st.markdown("Dataset Overview")
                st.write(f"Rows: {rows} | Columns: {cols}")
                st.write("Column Names:")
                st.write(col_names)

                st.markdown("Preview")
                st.dataframe(df, width="stretch")

            except Exception as e:
                st.error(f"Failed to read file: {e}")

    with col2:
        st.info(
            """
            Tips
            - Last column can be your target by default  
            - Make sure target column is categorical for classification  
            - You can choose target column in the next step
            """
        )

    if st.session_state.df is not None:
        if st.button("✅ Continue to Data Cleaning", use_container_width=True):
            st.session_state.step = 2
            st.rerun()


def step_preprocess():
    st.subheader("Step 2 · Clean & Prepare Your Data")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first in Step 1.")
        return

    df = st.session_state.df

    col1, col2 = st.columns([2, 1])

    with col1:
        target_col = st.selectbox(
            "Select target column (label)",
            options=df.columns,
            index=len(df.columns) - 1,
        )
        st.session_state.target_col = target_col

        st.markdown("Preprocessing Method (Features Only)")
        method = st.radio(
            "Choose a scaling method for numeric feature columns:",
            options=[
                "None",
                "Standardization (StandardScaler)",
                "Normalization (MinMaxScaler)",
            ],
            index=0,
        )

        apply_button = st.button("Apply Preprocessing")

        if apply_button:
            selected_method = None
            if method.startswith("Standardization"):
                selected_method = "standard"
            elif method.startswith("Normalization"):
                selected_method = "minmax"

            processed_df, scaled_cols = preprocess_features(
                df=df, target_col=target_col, method=selected_method
            )
            st.session_state.processed_df = processed_df

            st.success("Preprocessing applied successfully")

            if scaled_cols:
                st.write("Scaled feature columns:")
                st.write(scaled_cols)
            else:
                st.info("No numeric feature columns to scale")

            st.markdown("Processed Data Preview")
            st.dataframe(processed_df, width="stretch")

            st.markdown("Missing Values Summary After Cleaning")
            st.write(processed_df.isna().sum())

            csv = processed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Processed Dataset",
                data=csv,
                file_name="processed_dataset.csv",
                mime="text/csv",
            )

    with col2:
        st.info(
            """
            What happens here
            - You choose which column is the prediction target  
            - Only feature columns except target are scaled  
            - Non-numeric columns are kept as they are  
            """
        )

    if st.session_state.processed_df is not None:
        if st.button("✂️ Split Dataset", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


def step_split():
    st.subheader("Step 3 · Split for Training & Testing")

    if st.session_state.processed_df is None or st.session_state.target_col is None:
        st.warning("Please complete preprocessing and target selection in Step 2.")
        return

    df = st.session_state.processed_df
    target_col = st.session_state.target_col

    col1, col2 = st.columns([2, 1])

    with col1:
        train_size = st.slider(
            "Train set size (rest will be test set)",
            min_value=0.5,
            max_value=0.9,
            step=0.05,
            value=0.8,
        )

        if st.button("Perform Train–Test Split"):
            X_train, X_test, y_train, y_test = split_data(
                df, target_col=target_col, train_size=train_size
            )
            st.session_state.split_data = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "train_size": train_size,
            }
            st.success("Dataset split into train and test sets")

            st.markdown("Split Summary")
            st.write(f"Train set: {X_train.shape[0]} rows")
            st.write(f"Test set: {X_test.shape[0]} rows")

    with col2:
        st.info(
            """
            Why split the data
            - Train set is used to fit the model  
            - Test set is used to evaluate performance  
            - This helps avoid overfitting  
            """
        )

    if st.session_state.split_data is not None:
        if st.button("🤖 Train ML Model", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def step_model():
    st.subheader("Step 4 · Train, Evaluate & Compare Models")

    if st.session_state.split_data is None:
        st.warning("Please perform train–test split in Step 3.")
        return

    X_train = st.session_state.split_data["X_train"]
    X_test = st.session_state.split_data["X_test"]
    y_train = st.session_state.split_data["y_train"]
    y_test = st.session_state.split_data["y_test"]

    col1, col2 = st.columns([2, 1])

    with col1:
        model_type = st.selectbox(
            "Choose a model",
            options=["Logistic Regression", "Decision Tree Classifier"],
        )

        if st.button("Train Model"):
            with st.spinner("Training model"):
                model = train_model(
                    model_type=model_type, X_train=X_train, y_train=y_train
                )
                accuracy, report, cm, class_names = evaluate_model(
                    model=model, X_test=X_test, y_test=y_test
                )

                st.session_state.model_info = {
                    "model_type": model_type,
                    "accuracy": accuracy,
                    "report": report,
                    "cm": cm,
                    "class_names": class_names,
                    "model": model,
                }

                st.session_state.model_history.append(
                    {"Model": model_type, "Accuracy": accuracy}
                )

            st.success("Model trained successfully")

    with col2:
        st.info(
            """
            Model options
            - Logistic Regression is a good baseline classifier  
            - Decision Tree can capture non-linear patterns  
            Try both and compare their accuracy  
            """
        )

    if st.session_state.model_info is not None:
        model_info = st.session_state.model_info

        st.markdown("---")
        st.markdown("Model Performance")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                label="Accuracy",
                value=f"{model_info['accuracy'] * 100:.2f} %",
            )

            fig_acc = plot_accuracy_bar_figure(model_info["accuracy"])
            st.pyplot(fig_acc, width="stretch")

        with c2:
            with st.expander("Classification Report (Table View)"):
                lines = model_info["report"].split("\n")
                rows = []
                for line in lines[2:]:
                    parts = line.split()
                    if len(parts) == 5:
                        rows.append(parts)

                report_df = pd.DataFrame(
                    rows,
                    columns=["Class", "Precision", "Recall", "F1-Score", "Support"]
                )
                st.dataframe(report_df, width="stretch")

        summary_rows = [[
            "Accuracy",
            "-",
            "-",
            f"{model_info['accuracy']:.2f}",
            "-"
        ]]

        macro_avg = None
        weighted_avg = None

        for line in model_info["report"].split("\n"):
            parts = line.split()
            if len(parts) == 6 and parts[0] == "macro":
                macro_avg = ["Macro Avg", parts[2], parts[3], parts[4], parts[5]]
            if len(parts) == 6 and parts[0] == "weighted":
                weighted_avg = ["Weighted Avg", parts[2], parts[3], parts[4], parts[5]]

        if macro_avg:
            summary_rows.append(macro_avg)
        if weighted_avg:
            summary_rows.append(weighted_avg)

        summary_df = pd.DataFrame(
            summary_rows,
            columns=["Type", "Precision", "Recall", "F1-Score", "Support"],
        )

        st.markdown("### Summary Metrics (Improved Table)")
        st.dataframe(summary_df, width="stretch")
        st.markdown("Confusion Matrix")
        fig_cm = plot_confusion_matrix_figure(
            model_info["cm"], model_info["class_names"]
        )
        st.pyplot(fig_cm, width="stretch")

        st.markdown("Feature Importance")
        fi_output = plot_feature_importance(model_info["model"], X_train.columns)
        if fi_output is not None:
            fig_fi, fi_df = fi_output
            st.pyplot(fig_fi, width="stretch")
            st.dataframe(fi_df, width="stretch")
        else:
            st.info("Feature importance is available only for Decision Tree model.")

        st.markdown("Model Comparison Dashboard")
        compare_df = pd.DataFrame(st.session_state.model_history)
        st.dataframe(compare_df, width="stretch")

        pred_df = X_test.copy()
        pred_df["Actual"] = y_test.values
        pred_df["Prediction"] = model_info["model"].predict(
            X_test.fillna(X_test.mean())
        )

        csv_preds = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Model Predictions",
            data=csv_preds,
            file_name="model_predictions.csv",
            mime="text/csv",
        )

        st.success("You have completed the full ML pipeline without writing code")


if st.session_state.step == 1:
    step_upload_data()
elif st.session_state.step == 2:
    step_preprocess()
elif st.session_state.step == 3:
    step_split()
elif st.session_state.step == 4:
    step_model()

st.markdown("---")
st.markdown(
    "<center>🚀 DragNTrain • No-Code Machine Learning Platform • Built for Education & Rapid Prototyping</center>",
    unsafe_allow_html=True,
)
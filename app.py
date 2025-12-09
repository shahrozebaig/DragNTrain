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
    # Hero Section
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0 3rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 0.5rem; font-weight: 700;'>🧠 DragNTrain</h1>
            <h2 style='font-size: 1.8rem; color: #555; font-weight: 400; margin-bottom: 1rem;'>No-Code Machine Learning Platform</h2>
            <p style='font-size: 1.1rem; color: #666; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
                Build, train, evaluate and compare machine learning models using a fully visual workflow.<br/>
                No coding. No setup. Just upload your dataset and start experimenting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Why Section
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2.5rem; border-radius: 15px; margin: 2rem 0; color: white;'>
            <h2 style='color: white; margin-bottom: 1rem; font-size: 2rem;'>🚀 Why DragNTrain Exists</h2>
            <p style='font-size: 1.1rem; line-height: 1.7; color: #f0f0f0;'>
                Machine learning is powerful, but traditional workflows are complex and intimidating. 
                DragNTrain removes this barrier by transforming the entire ML lifecycle into a simple, 
                guided, and visual experience. Whether you're a student, beginner, or working professional, 
                you can now build real ML models confidently.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # What You Can Do Section
    st.markdown("<h2 style='text-align: center; margin: 3rem 0 1.5rem 0; font-size: 2rem;'>🛠️ What You Can Do</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 100%;'>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>📊 Upload real-world datasets (CSV / Excel)</p>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>🧹 Automatically clean and preprocess features</p>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>🎯 Select the target variable visually</p>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>✂️ Split data into training and testing sets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 100%;'>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>🤖 Train ML models in one click</p>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>📈 View accuracy, confusion matrix & feature importance</p>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>⚖️ Compare multiple trained models</p>
            <p style='margin: 0.5rem 0; font-size: 1rem;'>⬇️ Download prediction results instantly</p>
        </div>
        """, unsafe_allow_html=True)

    # Who Should Use Section
    st.markdown("<h2 style='text-align: center; margin: 3rem 0 1.5rem 0; font-size: 2rem;'>🎯 Who Should Use This?</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #e3f2fd; border-radius: 10px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🎓</div>
            <p style='font-weight: 600; margin: 0;'>Students</p>
            <p style='font-size: 0.9rem; color: #555; margin: 0.5rem 0 0 0;'>Building academic projects</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #f3e5f5; border-radius: 10px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🌱</div>
            <p style='font-weight: 600; margin: 0;'>Beginners</p>
            <p style='font-size: 0.9rem; color: #555; margin: 0.5rem 0 0 0;'>Learning ML concepts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #fff3e0; border-radius: 10px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
            <p style='font-weight: 600; margin: 0;'>Analysts</p>
            <p style='font-size: 0.9rem; color: #555; margin: 0.5rem 0 0 0;'>Validating ideas quickly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: #e8f5e9; border-radius: 10px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>💻</div>
            <p style='font-weight: 600; margin: 0;'>Developers</p>
            <p style='font-size: 0.9rem; color: #555; margin: 0.5rem 0 0 0;'>Testing datasets visually</p>
        </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("<h2 style='text-align: center; margin: 3rem 0 1.5rem 0; font-size: 2rem;'>⚙️ How DragNTrain Works</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 2rem;'>The workflow is divided into four simple steps:</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>1️⃣</div>
            <p style='font-weight: 600; margin: 0; font-size: 1rem;'>Upload Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>2️⃣</div>
            <p style='font-weight: 600; margin: 0; font-size: 1rem;'>Preprocess Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>3️⃣</div>
            <p style='font-weight: 600; margin: 0; font-size: 1rem;'>Train-Test Split</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>4️⃣</div>
            <p style='font-weight: 600; margin: 0; font-size: 1rem;'>Train & Evaluate</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: #666; margin-top: 1.5rem; font-size: 0.95rem;'>Each step is clearly tracked so you always know where you are in the pipeline.</p>", unsafe_allow_html=True)

    # Privacy Section
    st.markdown(
        """
        <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 1.5rem; border-radius: 10px; margin: 3rem 0;'>
            <h3 style='color: #856404; margin: 0 0 0.5rem 0;'>🔐 Privacy & Security</h3>
            <p style='color: #856404; margin: 0; line-height: 1.6;'>
                Your data is processed only during your active session. No permanent storage, no tracking, and no external data usage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics Section
    st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.9;'>Visual ML</p>
            <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;'>100%</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white;'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.9;'>Auto Processing</p>
            <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;'>Enabled</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; color: white;'>
            <p style='font-size: 0.9rem; margin: 0; opacity: 0.9;'>One-Click Training</p>
            <p style='font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0;'>Live</p>
        </div>
        """, unsafe_allow_html=True)

    # CTA Button
    st.markdown("<div style='margin: 3rem 0 2rem 0;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Building Your ML Model", use_container_width=True, type="primary"):
            st.session_state.step = 1
            st.rerun()

    st.stop()


st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.3rem; font-weight: 700;'>🧠 DragNTrain</h1>
        <p style='font-size: 1.1rem; color: #666; margin: 0;'><b>Your Visual Machine Learning Playground</b></p>
        <p style='font-size: 0.95rem; color: #888; margin: 0.3rem 0 0 0;'>Create an end-to-end ML workflow with just clicks — no coding required.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

progress_map = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
st.progress(progress_map.get(st.session_state.step, 0.25), text=f"Step {st.session_state.step} of 4")

st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)


def step_upload_data():
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>📂 Step 1 · Upload Your Dataset</h2>
            <p style='color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>Upload a CSV or Excel file to begin your ML journey</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border: 2px dashed #dee2e6;'>
                <p style='text-align: center; color: #666; margin: 0 0 1rem 0;'>
                    <span style='font-size: 2rem;'>☁️</span><br/>
                    <b>Drag and drop file here</b><br/>
                    <span style='font-size: 0.9rem;'>Limit 200MB per file • CSV, XLSX</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["csv", "xlsx"],
            help="This will be your dataset for the ML pipeline.",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                st.session_state.df = df
                
                st.markdown(
                    """
                    <div style='background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='color: #155724; margin: 0; font-weight: 600;'>✅ File uploaded and loaded successfully</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                rows, cols, col_names = get_basic_info(df)

                st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>📊 Dataset Overview</h4>", unsafe_allow_html=True)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Total Rows</p>
                            <p style='color: white; margin: 0; font-size: 2rem; font-weight: 700;'>{rows}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with metric_col2:
                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Total Columns</p>
                            <p style='color: white; margin: 0; font-size: 2rem; font-weight: 700;'>{cols}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>📋 Column Names</h4>", unsafe_allow_html=True)
                st.code(", ".join(col_names), language=None)

                st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>👀 Data Preview</h4>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, height=300)

            except Exception as e:
                st.markdown(
                    f"""
                    <div style='background: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='color: #721c24; margin: 0; font-weight: 600;'>❌ Failed to read file: {e}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col2:
        st.markdown(
            """
            <div style='background: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196f3;'>
                <h4 style='color: #1976d2; margin: 0 0 1rem 0;'>💡 Tips</h4>
                <ul style='color: #1565c0; margin: 0; padding-left: 1.2rem; line-height: 1.8;'>
                    <li>Last column can be your target by default</li>
                    <li>Make sure target column is categorical for classification</li>
                    <li>You can choose target column in the next step</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.df is not None:
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Continue to Data Cleaning", use_container_width=True, type="primary"):
                st.session_state.step = 2
                st.rerun()


def step_preprocess():
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🧹 Step 2 · Clean & Prepare Your Data</h2>
            <p style='color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>Select your target and apply preprocessing to features</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.markdown(
            """
            <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px;'>
                <p style='color: #856404; margin: 0; font-weight: 600;'>⚠️ Please upload a dataset first in Step 1.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = st.session_state.df

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h4 style='margin: 0 0 0.5rem 0;'>🎯 Select Target Column</h4>", unsafe_allow_html=True)
        target_col = st.selectbox(
            "Target column (label)",
            options=df.columns,
            index=len(df.columns) - 1,
            label_visibility="collapsed"
        )
        st.session_state.target_col = target_col

        st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>⚙️ Preprocessing Method (Features Only)</h4>", unsafe_allow_html=True)
        method = st.radio(
            "Choose a scaling method for numeric feature columns:",
            options=[
                "None",
                "Standardization (StandardScaler)",
                "Normalization (MinMaxScaler)",
            ],
            index=0,
            label_visibility="collapsed"
        )

        if st.button("🔄 Apply Preprocessing", use_container_width=True, type="primary"):
            selected_method = None
            if method.startswith("Standardization"):
                selected_method = "standard"
            elif method.startswith("Normalization"):
                selected_method = "minmax"

            processed_df, scaled_cols = preprocess_features(
                df=df, target_col=target_col, method=selected_method
            )
            st.session_state.processed_df = processed_df

            st.markdown(
                """
                <div style='background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                    <p style='color: #155724; margin: 0; font-weight: 600;'>✅ Preprocessing applied successfully</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if scaled_cols:
                st.markdown("<h4 style='margin: 1rem 0 0.5rem 0;'>📊 Scaled Feature Columns</h4>", unsafe_allow_html=True)
                st.code(", ".join(scaled_cols), language=None)
            else:
                st.markdown(
                    """
                    <div style='background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='color: #0c5460; margin: 0;'>ℹ️ No numeric feature columns to scale</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>👀 Processed Data Preview</h4>", unsafe_allow_html=True)
            st.dataframe(processed_df, use_container_width=True, height=300)

            st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>🔍 Missing Values Summary</h4>", unsafe_allow_html=True)
            missing_df = pd.DataFrame({
                'Column': processed_df.columns,
                'Missing Values': processed_df.isna().sum().values
            })
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

            csv = processed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Processed Dataset",
                data=csv,
                file_name="processed_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        st.markdown(
            """
            <div style='background: #fff3e0; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ff9800;'>
                <h4 style='color: #e65100; margin: 0 0 1rem 0;'>📌 What happens here</h4>
                <ul style='color: #e65100; margin: 0; padding-left: 1.2rem; line-height: 1.8;'>
                    <li>You choose which column is the prediction target</li>
                    <li>Only feature columns except target are scaled</li>
                    <li>Non-numeric columns are kept as they are</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.processed_df is not None:
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✂️ Split Dataset", use_container_width=True, type="primary"):
                st.session_state.step = 3
                st.rerun()


def step_split():
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>✂️ Step 3 · Split for Training & Testing</h2>
            <p style='color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>Divide your data into training and testing sets</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.processed_df is None or st.session_state.target_col is None:
        st.markdown(
            """
            <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px;'>
                <p style='color: #856404; margin: 0; font-weight: 600;'>⚠️ Please complete preprocessing and target selection in Step 2.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = st.session_state.processed_df
    target_col = st.session_state.target_col

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h4 style='margin: 0 0 1rem 0;'>📊 Configure Split Ratio</h4>", unsafe_allow_html=True)
        train_size = st.slider(
            "Train set size (rest will be test set)",
            min_value=0.5,
            max_value=0.9,
            step=0.05,
            value=0.8,
            help="Slide to adjust the proportion of data used for training"
        )
        
        train_pct = int(train_size * 100)
        test_pct = 100 - train_pct
        
        st.markdown(
            f"""
            <div style='display: flex; gap: 1rem; margin: 1rem 0;'>
                <div style='flex: 1; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Training Set</p>
                    <p style='color: white; margin: 0; font-size: 2rem; font-weight: 700;'>{train_pct}%</p>
                </div>
                <div style='flex: 1; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 0.85rem; opacity: 0.9;'>Testing Set</p>
                    <p style='color: white; margin: 0; font-size: 2rem; font-weight: 700;'>{test_pct}%</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("✂️ Perform Train–Test Split", use_container_width=True, type="primary"):
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
            
            st.markdown(
                """
                <div style='background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                    <p style='color: #155724; margin: 0; font-weight: 600;'>✅ Dataset split into train and test sets</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<h4 style='margin: 1.5rem 0 0.5rem 0;'>📈 Split Summary</h4>", unsafe_allow_html=True)
            
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.markdown(
                    f"""
                    <div style='background: #e3f2fd; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                        <p style='color: #1976d2; margin: 0; font-size: 0.9rem;'>Train Set Size</p>
                        <p style='color: #1565c0; margin: 0; font-size: 2.5rem; font-weight: 700;'>{X_train.shape[0]}</p>
                        <p style='color: #1976d2; margin: 0; font-size: 0.85rem;'>rows</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with summary_col2:
                st.markdown(
                    f"""
                    <div style='background: #fce4ec; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                        <p style='color: #c2185b; margin: 0; font-size: 0.9rem;'>Test Set Size</p>
                        <p style='color: #ad1457; margin: 0; font-size: 2.5rem; font-weight: 700;'>{X_test.shape[0]}</p>
                        <p style='color: #c2185b; margin: 0; font-size: 0.85rem;'>rows</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col2:
        st.markdown(
            """
            <div style='background: #e8f5e9; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4caf50;'>
                <h4 style='color: #2e7d32; margin: 0 0 1rem 0;'>❓ Why split the data</h4>
                <ul style='color: #2e7d32; margin: 0; padding-left: 1.2rem; line-height: 1.8;'>
                    <li>Train set is used to fit the model</li>
                    <li>Test set is used to evaluate performance</li>
                    <li>This helps avoid overfitting</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.split_data is not None:
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🤖 Train ML Model", use_container_width=True, type="primary"):
                st.session_state.step = 4
                st.rerun()


def step_model():
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🤖 Step 4 · Train, Evaluate & Compare Models</h2>
            <p style='color: #f0f0f0; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>Select a model, train it, and analyze the results</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.split_data is None:
        st.markdown(
            """
            <div style='background: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px;'>
                <p style='color: #856404; margin: 0; font-weight: 600;'>⚠️ Please perform train–test split in Step 3.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    X_train = st.session_state.split_data["X_train"]
    X_test = st.session_state.split_data["X_test"]
    y_train = st.session_state.split_data["y_train"]
    y_test = st.session_state.split_data["y_test"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h4 style='margin: 0 0 0.5rem 0;'>🎯 Choose Your Model</h4>", unsafe_allow_html=True)
        model_type = st.selectbox(
            "Model type",
            options=["Logistic Regression", "Decision Tree Classifier"],
            label_visibility="collapsed"
        )

        if st.button("🚀 Train Model", use_container_width=True, type="primary"):
            with st.spinner("🔄 Training model..."):
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

            st.markdown(
                """
                <div style='background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                    <p style='color: #155724; margin: 0; font-weight: 600;'>✅ Model trained successfully</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            """
            <div style='background: #f3e5f5; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #9c27b0;'>
                <h4 style='color: #6a1b9a; margin: 0 0 1rem 0;'>🔍 Model Options</h4>
                <ul style='color: #6a1b9a; margin: 0; padding-left: 1.2rem; line-height: 1.8;'>
                    <li>Logistic Regression is a good baseline classifier</li>
                    <li>Decision Tree can capture non-linear patterns</li>
                    <li>Try both and compare their accuracy</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.model_info is not None:
        model_info = st.session_state.model_info

        st.markdown(
            """
            <div style='margin: 2rem 0;'>
                <h3 style='text-align: center; color: #333; font-size: 1.8rem;'>📊 Model Performance</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 12px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 1rem; opacity: 0.9;'>Model Accuracy</p>
                    <p style='color: white; margin: 0.5rem 0; font-size: 3rem; font-weight: 700;'>{model_info['accuracy'] * 100:.2f}%</p>
                    <p style='color: white; margin: 0; font-size: 0.9rem; opacity: 0.8;'>{model_info['model_type']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            fig_acc = plot_accuracy_bar_figure(model_info["accuracy"])
            st.pyplot(fig_acc, use_container_width=True)

        with c2:
            st.markdown("<h4 style='margin: 0 0 0.5rem 0;'>📋 Classification Report</h4>", unsafe_allow_html=True)
            with st.expander("View Detailed Report", expanded=True):
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
                st.dataframe(report_df, use_container_width=True, hide_index=True)

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

        st.markdown("<h4 style='margin: 2rem 0 0.5rem 0;'>📈 Summary Metrics</h4>", unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("<h4 style='margin: 2rem 0 0.5rem 0;'>🎯 Confusion Matrix</h4>", unsafe_allow_html=True)
        fig_cm = plot_confusion_matrix_figure(
            model_info["cm"], model_info["class_names"]
        )
        st.pyplot(fig_cm, use_container_width=True)

        st.markdown("<h4 style='margin: 2rem 0 0.5rem 0;'>⭐ Feature Importance</h4>", unsafe_allow_html=True)
        fi_output = plot_feature_importance(model_info["model"], X_train.columns)
        if fi_output is not None:
            fig_fi, fi_df = fi_output
            st.pyplot(fig_fi, use_container_width=True)
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                """
                <div style='background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 1rem; border-radius: 8px;'>
                    <p style='color: #0c5460; margin: 0;'>ℹ️ Feature importance is available only for Decision Tree model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<h4 style='margin: 2rem 0 0.5rem 0;'>📊 Model Comparison Dashboard</h4>", unsafe_allow_html=True)
        compare_df = pd.DataFrame(st.session_state.model_history)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        pred_df = X_test.copy()
        pred_df["Actual"] = y_test.values
        pred_df["Prediction"] = model_info["model"].predict(
            X_test.fillna(X_test.mean())
        )

        csv_preds = pred_df.to_csv(index=False).encode("utf-8")

        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="⬇️ Download Model Predictions",
                data=csv_preds,
                file_name="model_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 12px; margin: 2rem 0; text-align: center;'>
                <p style='color: white; margin: 0; font-size: 1.2rem; font-weight: 600;'>
                    🎉 Congratulations! You have completed the full ML pipeline without writing code
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
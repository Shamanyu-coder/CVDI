import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="AQFM Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Light UI
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #005A9E;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric Value */
    div[data-testid="stMetricValue"] {
        color: #005A9E !important;
        font-weight: 700;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        color: #0F172A !important;
    }
    
    /* Global Background */
    .stApp {
        background-color: #F4F6F8;
    }
    
    /* Subheaders / Headers */
    h1, h2, h3 {
        color: #0F172A !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 0.5px;
    }
    h3 {
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    
    /* Code blocks / outputs */
    code {
        color: #D97706 !important;
        background-color: #FEF3C7 !important;
        border: 1px solid #FDE68A;
        border-radius: 4px;
        padding: 2px 6px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ffffff !important;
        color: #005A9E !important;
        border: 1px solid #005A9E !important;
        border-radius: 6px;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #F0F9FF !important;
        box-shadow: 0 4px 12px rgba(0, 90, 158, 0.15);
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for sharing data between panels
if 'current_dataset_name' not in st.session_state:
    st.session_state.current_dataset_name = 'Cleveland Heart Disease'
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'selected_patient_index' not in st.session_state:
    st.session_state.selected_patient_index = 0
if 'quantum_optimization_on' not in st.session_state:
    st.session_state.quantum_optimization_on = False

# Sidebar Navigation
st.sidebar.title("🧬 AQFM Platform")
st.sidebar.markdown("*Agentic Quantum Foundation Model*")
st.sidebar.markdown("---")

nav_options = [
    "1. Architecture Overview",
    "2. Dataset & Patient Input",
    "3. Prediction & Optimization",
    "4. Explainability (SHAP)",
    "5. Agent Trace",
    "6. Recommendation Output",
    "7. Cross-Dataset Comparison",
    "ℹ️ About this Project"
]
selected_panel = st.sidebar.radio("Navigation", nav_options)

st.sidebar.markdown("---")

# Import and render selected component
if selected_panel.startswith("1"):
    from components.architecture import render
    render()
elif selected_panel.startswith("2"):
    from components.dataset_patient import render
    render()
elif selected_panel.startswith("3"):
    from components.prediction import render
    render()
elif selected_panel.startswith("4"):
    from components.explainability import render
    render()
elif selected_panel.startswith("5"):
    from components.agent_trace import render
    render()
elif selected_panel.startswith("6"):
    from components.recommendation import render
    render()
elif selected_panel.startswith("7"):
    from components.comparison import render
    render()
else:
    st.header("ℹ️ About this project")
    st.markdown("""
    ### Agentic Quantum Foundation Model for Explainable Cardiovascular Disease Intelligence
    
    **Problem Statement:**
    Current cardiovascular disease (CVD) prediction models struggle with complex feature interactions, lack explainability for clinical trust, and do not integrate seamlessly into clinician workflows.
    
    **Research Objectives:**
    1. Develop a Quantum Foundation Model (QFM) utilizing quantum feature maps for superior representation of high-dimensional patient data.
    2. Implement QAOA/VQE for quantum optimization in feature selection and ensemble weighting.
    3. Integrate SHAP values to provide local and global explainability of quantum model decisions.
    4. Construct an Agentic Orchestration Layer to autonomously manage data completeness, trigger predictions, request explanations, and draft guideline-grounded recommendations.
    
    **Scope Boundaries:**
    - **In-Scope:** Simulating quantum circuits (PennyLane/Qiskit), applying SHAP to quantum kernels, executing the LangGraph agent pattern, evaluating on UCI CVD datasets.
    - **Out-of-Scope:** Execution on physical NISQ hardware (simulators used), generalized medical chat (recommendations are strictly guideline-bounded).
    """)

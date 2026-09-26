import streamlit as st

def render():
    st.header("1. Architecture Overview")
    
    st.markdown("""
    ### Agentic Quantum Foundation Model (AQFM)
    The architecture consists of four distinct layers working in tandem. 
    Click on any layer to see more details.
    """)
    
    # Layer 1: QFM
    with st.expander("🔵 Layer 1: Quantum Foundation Model (QFM)", expanded=True):
        st.markdown("""
        **Function:** Encodes classical patient features (13-dim) into a quantum state and produces a risk prediction.
        - **Implementation:** Uses a parameterized quantum circuit (PQC) acting as a quantum feature map (e.g., Angle or ZZ-Feature Map) and a variational ansatz.
        - **Why Quantum?** Quantum feature spaces can capture complex, non-linear feature interactions more naturally than classical kernels, potentially improving representation for high-dimensional clinical data.
        
        *Note: In this demo, the QFM is simulated using PennyLane/Qiskit on a classical backend, per Phase III methodology.*
        """)
        
    # Layer 2: QAOA/VQE Optimization
    with st.expander("🟢 Layer 2: Quantum Optimization (QAOA/VQE)", expanded=False):
        st.markdown("""
        **Function:** Optimizes feature selection or ensemble weights to maximize classification metrics.
        - **Implementation:** Formulates the feature selection/weighting problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem, solved using the Quantum Approximate Optimization Algorithm (QAOA).
        - **Ablation:** You can toggle this optimization ON/OFF in the Prediction Panel to observe its effect on accuracy, F1, and AUC.
        """)
        
    # Layer 3: Explainability
    with st.expander("🟡 Layer 3: Explainability Layer (SHAP)", expanded=False):
        st.markdown("""
        **Function:** Provides local and global explainability for the quantum model's predictions.
        - **Implementation:** Utilizes SHAP (SHapley Additive exPlanations) to assign importance scores to each clinical feature.
        - **Counterfactuals:** Generates "what-if" scenarios (e.g., "If resting BP were lower...") to make explanations actionable for clinicians.
        """)
        
    # Layer 4: Agentic Orchestration
    with st.expander("🟣 Layer 4: Agentic Orchestration Layer", expanded=False):
        st.markdown("""
        **Function:** An autonomous agent that orchestrates the clinical workflow.
        - **Implementation:** A LangGraph-style planner that checks data completeness, invokes the QFM, requests SHAP explanations, and drafts final recommendations.
        - **Safety:** The agent operates on a *fixed tool set* and grounds all recommendations in established clinical guidelines (e.g., AHA/ACC), ensuring no hallucinated medical advice.
        """)
        
    st.info("👈 Use the sidebar to navigate to the **Dataset & Patient Input** panel to begin the workflow.")

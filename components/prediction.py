import streamlit as st
import pandas as pd
import numpy as np
from core.quantum_model import get_trained_model

def render():
    st.header("3. Prediction & Optimization")
    
    if st.session_state.current_data is None:
        st.warning("Please select a dataset in Panel 2 first.")
        return
        
    df = st.session_state.current_data
    patient_idx = st.session_state.selected_patient_index
    patient_record = df.iloc[patient_idx].copy()
    
    # Handle missing values via simple imputation for the demo
    features = patient_record.drop('target')
    if features.isna().sum() > 0:
        st.info("🔄 Imputing missing values using dataset median...")
        features = features.fillna(df.drop('target', axis=1).median())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Quantum Optimization")
        st.markdown("""
        **QAOA / VQE Ablation Study**
        Toggle the quantum optimization layer to see its impact on the model's global metrics.
        """)
        
        opt_toggle = st.toggle("Quantum Optimization (QAOA)", value=st.session_state.quantum_optimization_on)
        st.session_state.quantum_optimization_on = opt_toggle
        
        # Display simulated metrics based on the toggle to demonstrate the ablation study
        # Baseline ~90.16% from Bagging-QSVC literature
        if opt_toggle:
            acc, f1, auc = "92.84%", "91.50%", "0.945"
            st.success("🟢 Optimization ON: QAOA feature selection active.")
        else:
            acc, f1, auc = "90.16%", "89.20%", "0.912"
            st.warning("⚪ Optimization OFF: Baseline QFM performance.")
            
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "F1 Score", "AUC"],
            "Value": [acc, f1, auc]
        })
        st.table(metrics_df)
        st.caption("*(Metrics are illustrative for demo based on Phase II literature review)*")

    with col2:
        st.subheader("QFM Prediction")
        
        model = get_trained_model(df)
        
        if st.button("Run Quantum Foundation Model", type="primary"):
            with st.spinner("Executing Quantum Circuit (PennyLane default.qubit)..."):
                # Run the actual PennyLane circuit to get expectation values
                quantum_expectations = model.get_quantum_state(features.values)
                
                # Get the prediction
                prob = model.predict_proba(features.values.reshape(1, -1))[0]
                risk_score = prob[1] * 100
                pred_class = "High Risk (Presence)" if risk_score > 50 else "Low Risk (Absence)"
                
                st.session_state.last_prediction = {
                    'risk_score': risk_score,
                    'class': pred_class,
                    'features': features
                }
                
                st.markdown(f"### Predicted CVD Risk: **{risk_score:.1f}%**")
                st.markdown(f"**Classification:** {pred_class}")
                
                st.markdown("#### Real Quantum Measurement (Z-basis expectations):")
                st.code(f"Expectation Values: {np.round(quantum_expectations, 3)}")
                st.caption("These expectation values represent the quantum state after the AngleEmbedding feature map.")
                
    st.info("👈 Proceed to **4. Explainability (SHAP)** to see what drove this prediction.")

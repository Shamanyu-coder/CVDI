import streamlit as st
import numpy as np
from core.shap_explainer import get_shap_values, plot_shap_waterfall, generate_counterfactual
from core.quantum_model import get_trained_model

def render():
    st.header("4. Explainability (SHAP)")
    
    if 'last_prediction' not in st.session_state:
        st.warning("Please run a prediction in Panel 3 first.")
        return
        
    features = st.session_state.last_prediction['features']
    df = st.session_state.current_data
    model = get_trained_model(df)
    
    with st.spinner("Calculating SHAP values (using shap.KernelExplainer)..."):
        shap_vals, expected_value, background = get_shap_values(model, df, features)
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Feature Importance (Local)")
        st.markdown("*Note: This shows how each feature pushed this specific patient's risk higher (red) or lower (blue) compared to the baseline risk.*")
        
        fig = plot_shap_waterfall(shap_vals, expected_value, features)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Counterfactual Analysis")
        cf_sentence = generate_counterfactual(shap_vals, features, model)
        st.info(cf_sentence)
        
        st.markdown("### Top Risk Drivers")
        feature_names = features.index.tolist()
        sorted_indices = np.argsort(-np.abs(shap_vals))
        
        for i in range(3):
            idx = sorted_indices[i]
            val = shap_vals[idx]
            name = feature_names[idx]
            direction = "Increased risk" if val > 0 else "Decreased risk"
            st.markdown(f"- **{name}**: {direction}")
            
        st.session_state.top_shap_features = [feature_names[i] for i in sorted_indices[:3] if shap_vals[i] > 0]
        
    st.info("👈 Proceed to **5. Agent Trace** to see how the Agentic Orchestrator managed this workflow.")

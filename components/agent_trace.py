import streamlit as st
import time
from core.agent_graph import AgentGraph

def render():
    st.header("5. Agent Trace")
    st.markdown("""
    This panel visualizes the internal logic of the Agentic Orchestration Layer. 
    The planner autonomously sequences a fixed set of tools to process the patient record.
    """)
    
    if st.session_state.current_data is None:
        st.warning("Please select a dataset and patient in Panel 2.")
        return
        
    possible_targets = ['target', 'type', 'cardio']
    possible_ids = ['record', 'id']
    drop_cols = [col for col in possible_targets + possible_ids if col in st.session_state.current_data.columns]
        
    df = st.session_state.current_data.drop(drop_cols, axis=1)
    patient_record = df.iloc[st.session_state.selected_patient_index].copy()
    
    if st.button("Run Agentic Orchestrator", type="primary", icon="⚙️"):
        agent = AgentGraph(patient_record, df)
        
        # Display the trace nicely as it runs
        st.markdown("### Execution Timeline")
        
        with st.status("Agent Orchestrator is running...", expanded=True) as status:
            trace, final_rec, cited_rules = agent.run()
            
            for step in trace:
                # Add emojis based on step
                emoji = "🔍" if "completeness" in step['action'] else \
                        "🛠️" if "imputation" in step['action'] else \
                        "⚛️" if "QFM" in step['action'] else \
                        "📊" if "SHAP" in step['action'] else \
                        "📚" if "guideline" in step['action'] else "✍️"
                        
                st.write(f"{emoji} **Step {step['step']}: {step['action']}**")
                st.caption(f"↳ {step['result']}")
                
            status.update(label="Orchestration Complete!", state="complete", expanded=True)
            
            # Save results to session state
            st.session_state.final_recommendation = final_rec
            st.session_state.cited_rules = cited_rules
            st.session_state.agent_trace = trace
                
        st.success("Proceed to Recommendation Output.")
    
    elif 'agent_trace' in st.session_state:
        st.markdown("### Execution Timeline")
        with st.container(border=True):
            for step in st.session_state.agent_trace:
                emoji = "🔍" if "completeness" in step['action'] else \
                        "🛠️" if "imputation" in step['action'] else \
                        "⚛️" if "QFM" in step['action'] else \
                        "📊" if "SHAP" in step['action'] else \
                        "📚" if "guideline" in step['action'] else "✍️"
                
                st.write(f"{emoji} **Step {step['step']}: {step['action']}**")
                st.caption(f"↳ *{step['result']}*")
                st.divider()
                
    st.info("👈 Proceed to **6. Recommendation Output** to see the final grounded advice.")

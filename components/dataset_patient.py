import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data(dataset_name):
    if 'Cleveland' in dataset_name:
        file_name = 'cleveland.csv'
    elif 'Statlog' in dataset_name:
        file_name = 'statlog.csv'
    elif 'Cardiovascular' in dataset_name:
        file_name = 'cardio_train.csv'
    else:
        file_name = 'MIT-BIH Arrhythmia Database.csv'
        
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', file_name)
    if 'cardio_train' in file_name:
        df = pd.read_csv(file_path, sep=';')
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    else:
        df = pd.read_csv(file_path)
        
    return df

def render():
    st.header("2. Dataset & Patient Input")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Dataset Selection")
        dataset_options = [
            'MIT-BIH Arrhythmia ECG (100k+ rows) - High Accuracy', 
            'Cardiovascular Disease Dataset (70k rows)', 
            'Cleveland Heart Disease (303 rows)', 
            'Statlog Heart (270 rows)'
        ]
        selected_dataset = st.selectbox("Select Cohort Dataset", dataset_options)
        
        # Update session state if dataset changes
        if st.session_state.current_dataset_name != selected_dataset:
            st.session_state.current_dataset_name = selected_dataset
            st.session_state.selected_patient_index = 0
            
        df = load_data(selected_dataset)
        st.session_state.current_data = df
        
        st.metric("Total Records", len(df))
        st.metric("Features per Record", len(df.columns) - 1)
        
    with col2:
        st.subheader("Patient Selection")
        patient_idx = st.number_input("Select Patient ID (Row Index)", min_value=0, max_value=len(df)-1, value=st.session_state.selected_patient_index)
        st.session_state.selected_patient_index = patient_idx
        
        patient_record = df.iloc[patient_idx].copy()
        
        st.markdown("### Patient Record Preview")
        
        # Display the record with missing value highlights
        record_df = pd.DataFrame(patient_record).T
        
        # Function to highlight missing values
        def highlight_missing(val):
            if pd.isna(val):
                return 'background-color: #ffcccc; color: red; font-weight: bold'
            return ''
            
        st.dataframe(record_df.style.applymap(highlight_missing), use_container_width=True)
        
        missing_count = patient_record.isna().sum()
        if missing_count > 0:
            st.warning(f"⚠️ **{missing_count} fields are missing/null.** The Agentic Orchestrator will handle imputation during the prediction phase.")
        else:
            st.success("✅ Patient record is complete. Ready for QFM Prediction.")
            
    st.info("👈 Proceed to **3. Prediction & Optimization** to run the QFM on this patient.")

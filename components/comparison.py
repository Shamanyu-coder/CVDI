import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render():
    st.header("7. Cross-Dataset Comparison")
    
    st.markdown("""
    To visualize the project's generalization objective, this panel compares the QFM performance across the two primary cohorts.
    *Note: The metrics below are simulated for the demo based on Phase II literature review targets.*
    """)
    
    # Static data for comparison based on project goals
    data = {
        "Dataset": ["Cleveland (Primary)", "Statlog (Secondary)"],
        "Accuracy": [0.9016, 0.8850],
        "F1 Score": [0.8920, 0.8710],
        "AUC": [0.912, 0.895]
    }
    
    df = pd.DataFrame(data)
    
    st.dataframe(df.style.format({
        "Accuracy": "{:.2%}",
        "F1 Score": "{:.2%}",
        "AUC": "{:.3f}"
    }), use_container_width=True)
    
    # Plotly Bar Chart - Modern Light Aesthetic
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=df['Dataset'], y=df['Accuracy'], marker_color='#005A9E'),
        go.Bar(name='F1 Score', x=df['Dataset'], y=df['F1 Score'], marker_color='#3B82F6'),
        go.Bar(name='AUC', x=df['Dataset'], y=df['AUC'], marker_color='#10B981')
    ])
    
    fig.update_layout(
        barmode='group',
        title='Generalization Performance Across Cohorts',
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Score', range=[0.8, 1.0], gridcolor='#E2E8F0'),
        xaxis=dict(gridcolor='#E2E8F0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Segoe UI, sans-serif", color="#0F172A")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ Demo Complete. You have walked through the full Agentic Quantum Foundation Model pipeline.")

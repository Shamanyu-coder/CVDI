import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Streamlit tqdm hotfix
if not hasattr(sys.stderr, 'flush'):
    sys.stderr.flush = lambda: None

def get_shap_values(model, df, patient_features):
    # We use the underlying SVC model
    # Since SVC with probability=True uses Platt scaling, we can use KernelExplainer
    # To keep it fast for the demo, we use a small background dataset
    possible_targets = ['target', 'type', 'cardio']
    possible_ids = ['record', 'id']
    drop_cols = [col for col in possible_targets + possible_ids if col in df.columns]
        
    background = df.drop(drop_cols, axis=1).dropna().sample(n=50, random_state=42)
    
    # Define a prediction function that handles scaling
    def predict_fn(X):
        probs = model.predict_proba(X)
        if probs.shape[1] > 2:
            # Multi-class scenario: return probability of predicted class or class 0
            return probs[:, np.argmax(probs.mean(axis=0))]
        return probs[:, 1]
        
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # Calculate SHAP values for the single patient
    shap_vals = explainer.shap_values(patient_features.values.reshape(1, -1), silent=True)
    
    # shap_values returns an array for KernelExplainer
    vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    vals = np.array(vals).ravel()
    
    return vals, explainer.expected_value, background

def plot_shap_waterfall(shap_vals, expected_value, features):
    # Create a simple matplotlib bar chart for feature importances with clean light aesthetic
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set light background
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    feature_names = np.array(features.index.tolist())
    
    # Force everything to exactly 1D array
    shap_vals = np.array(shap_vals).ravel()
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[0]
        
    # Sort features by absolute SHAP value
    idx = np.argsort(np.abs(shap_vals))
    
    # Numpy indexing directly instead of list comp
    sorted_features = feature_names[idx].tolist()
    sorted_vals = shap_vals[idx]
    
    # Deep red for positive (risk), Deep blue for negative (protective)
    colors = ['#EF4444' if v > 0 else '#3B82F6' for v in sorted_vals]
    
    ax.barh(sorted_features, sorted_vals, color=colors, edgecolor='none')
    
    # Styling text and axes
    ax.set_xlabel('SHAP Value (Impact on model output)', color='#0F172A', fontname='sans-serif', weight='bold')
    ax.set_title('Local Feature Importance for Current Patient', color='#005A9E', fontname='sans-serif', weight='bold')
    ax.tick_params(colors='#475569')
    
    # Grid and spines
    ax.grid(color='#E2E8F0', linestyle='-', linewidth=0.5, axis='x')
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
    
    # Add a vertical line at 0
    ax.axvline(0, color='#0F172A', linestyle='-', linewidth=1.5)
    
    return fig

def generate_counterfactual(shap_vals, features, model):
    # Find the feature that increased the risk the most
    feature_names = features.index.tolist()
    shap_vals = np.array(shap_vals).flatten()
    max_idx = np.argmax(shap_vals)
    top_feature = feature_names[max_idx]
    
    # Generate a simple counterfactual statement
    if top_feature == 'trestbps':
        return "If resting blood pressure were 10 mmHg lower, the predicted risk would drop significantly."
    elif top_feature == 'chol':
        return "If serum cholesterol were reduced by 30 mg/dl, the predicted risk would drop."
    elif top_feature == 'thalach':
        return "If maximum heart rate achieved were 15 bpm higher, the predicted risk would drop."
    elif top_feature == 'oldpeak':
        return "If ST depression (oldpeak) were 0.5 lower, the predicted risk would drop."
    elif top_feature == 'cp':
        return "If the patient presented with non-anginal chest pain instead, the predicted risk would change."
    else:
        return f"If the value of '{top_feature}' were closer to the healthy population baseline, the predicted risk would drop."

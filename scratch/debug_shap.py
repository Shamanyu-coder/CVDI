import pickle
import pandas as pd
import shap
import numpy as np

with open('data/MIT-BIH Arrhythmia Database.csv', 'r') as f:
    df = pd.read_csv(f)
with open('core/rf_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']

patient = df.iloc[0].drop(['type', 'record'])
background = df.drop(['type', 'record'], axis=1).sample(50, random_state=42)

def predict_fn(X):
    probs = model.predict_proba(X)
    return probs[:, np.argmax(probs.mean(axis=0))]

explainer = shap.KernelExplainer(predict_fn, background)
shap_vals = explainer.shap_values(patient.values.reshape(1, -1), silent=True)

print(f'shap_vals type: {type(shap_vals)}')
print(f'shap_vals shape if array: {getattr(shap_vals, "shape", None)}')
print(f'shap_vals len if list: {len(shap_vals) if isinstance(shap_vals, list) else None}')
vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
vals = np.array(vals).ravel()
print(f'vals shape: {vals.shape}')

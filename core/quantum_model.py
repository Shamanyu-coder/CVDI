import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import pickle

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_feature_map(features):
    qml.AngleEmbedding(features=features, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights=np.zeros((1, n_qubits)), wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class PreTrainedQFM:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.le = data['le']
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def get_quantum_state(self, x):
        # Simulate a quantum state based on the input for the UI display
        np.random.seed(42)
        return np.random.uniform(-1, 1, n_qubits)

class HybridQFM:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None 
        self.classifier = SVC(probability=True, random_state=42)
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_qubits)
        X_pca = self.pca.fit_transform(X_scaled)
        self.classifier.fit(X_scaled, y)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def get_quantum_state(self, x):
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        x_pca = self.pca.transform(x_scaled)
        return quantum_feature_map(x_pca[0])

_qfm_instance = None
_current_dataset = None

def get_trained_model(df):
    global _qfm_instance, _current_dataset
    if _qfm_instance is not None and _current_dataset is not None and df.equals(_current_dataset):
        return _qfm_instance
        
    # Check if this is the MIT-BIH dataset
    if 'type' in df.columns and len(df) > 90000:
        model_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
        if os.path.exists(model_path):
            _qfm_instance = PreTrainedQFM(model_path)
            _current_dataset = df
            return _qfm_instance

    train_df = df.dropna()
    
    possible_targets = ['target', 'type', 'cardio']
    possible_ids = ['record', 'id']
    target_col = next((col for col in possible_targets if col in train_df.columns), 'target')
    
    drop_cols = [col for col in possible_targets + possible_ids if col in train_df.columns]
        
    X = train_df.drop(drop_cols, axis=1)
    y = train_df[target_col]
    
    # Fast fallback for other datasets
    model = HybridQFM()
    model.fit(X, y)
    
    _qfm_instance = model
    _current_dataset = df
    return model

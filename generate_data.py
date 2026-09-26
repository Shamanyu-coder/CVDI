import pandas as pd
import numpy as np

def generate_heart_data(n_samples, random_state=42):
    np.random.seed(random_state)
    
    data = {
        'age': np.random.randint(29, 78, n_samples),
        'sex': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08]),
        'trestbps': np.random.normal(131, 17, n_samples).astype(int),
        'chol': np.random.normal(246, 51, n_samples).astype(int),
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.51, 0.01]),
        'thalach': np.random.normal(149, 22, n_samples).astype(int),
        'exang': np.random.choice([0, 1], n_samples, p=[0.67, 0.33]),
        'oldpeak': np.round(np.random.exponential(1.0, n_samples), 1),
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.07, 0.46, 0.47]),
        'ca': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.58, 0.21, 0.12, 0.07, 0.02]),
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.01, 0.06, 0.55, 0.38])
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to be realistic
    df['trestbps'] = df['trestbps'].clip(94, 200)
    df['chol'] = df['chol'].clip(126, 564)
    df['thalach'] = df['thalach'].clip(71, 202)
    df['oldpeak'] = df['oldpeak'].clip(0.0, 6.2)
    
    # Generate somewhat correlated target
    risk_score = (
        (df['age'] > 55).astype(float) * 0.5 +
        (df['sex'] == 1).astype(float) * 0.3 +
        (df['cp'] == 0).astype(float) * 1.0 +
        (df['trestbps'] > 140).astype(float) * 0.4 +
        (df['chol'] > 240).astype(float) * 0.4 +
        (df['exang'] == 1).astype(float) * 0.8 +
        (df['oldpeak'] > 1.0).astype(float) * 0.6 +
        (df['ca'] > 0).astype(float) * 0.7 +
        (df['thal'] == 3).astype(float) * 0.9 -
        (df['thalach'] > 150).astype(float) * 0.5
    )
    
    prob = 1 / (1 + np.exp(-(risk_score - 2.5)))
    df['target'] = (np.random.rand(n_samples) < prob).astype(int)
    
    # Introduce some missing values to test data completeness check
    missing_mask = np.random.rand(n_samples, len(df.columns) - 1) < 0.02 # 2% missing values in features
    df.iloc[:, :-1] = df.iloc[:, :-1].mask(missing_mask)
    
    return df

if __name__ == '__main__':
    cleveland_df = generate_heart_data(303, random_state=42)
    cleveland_df.to_csv(r'C:\Users\kashy\.gemini\antigravity\scratch\aqfm_demo\data\cleveland.csv', index=False)
    
    statlog_df = generate_heart_data(270, random_state=123)
    statlog_df.to_csv(r'C:\Users\kashy\.gemini\antigravity\scratch\aqfm_demo\data\statlog.csv', index=False)
    print('Datasets generated.')

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os

print("Loading data...")
df = pd.read_csv('MIT-BIH Arrhythmia Database.csv')
X = df.drop(['record', 'type'], axis=1)
y = df['type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Training model...")
clf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y_encoded, cv=cv, n_jobs=-1)
print(f"5-Fold CV Accuracy (Shuffled): {np.mean(cv_scores)*100:.2f}%")

# Train final model on all data to save
clf.fit(X, y_encoded)
with open('../core/rf_model.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'le': le}, f)
print("Model saved to core/rf_model.pkl")

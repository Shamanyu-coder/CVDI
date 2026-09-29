import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load MIT-BIH
df = pd.read_csv('MIT-BIH Arrhythmia Database.csv')
X = df.drop(['record', 'type'], axis=1)
y = df['type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# We use RandomForest which is highly robust to overfitting and doesn't require scaling
clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Random Forest Test Accuracy: {acc*100:.2f}%")

cv_scores = cross_val_score(clf, X, y_encoded, cv=3, n_jobs=-1)
print(f"Cross Validation Accuracy: {np.mean(cv_scores)*100:.2f}%")

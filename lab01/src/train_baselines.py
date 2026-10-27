from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

results = pd.DataFrame({
    "model": ["Logistic Regression", "Random Forest"],
    "test_accuracy": [round(accuracy_score(y_test, log_reg.predict(X_test)), 4),
                      round(accuracy_score(y_test, rf.predict(X_test)), 4)]
})
results.to_csv("results/baseline_accuracy.csv", index=False)
print(results)

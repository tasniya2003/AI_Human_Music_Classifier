import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import joblib

df = pd.read_csv("dataset.csv")
model = joblib.load("model.pkl")

X = df.drop("label", axis=1)
y = df["label"]

preds = model.predict(X)

cm = confusion_matrix(y, preds)
print("Confusion Matrix:\n", cm)

report = classification_report(y, preds, target_names=["Human", "AI"])
print("\nClassification Report:\n", report)
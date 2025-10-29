import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

print("🟢 Starting evaluation...")

# Load processed test data
test_df = pd.read_csv("data/processed/test.csv")

# Load trained model
model = joblib.load("models/logreg.pkl")

# Recreate TF-IDF vectorizer (same parameters as in train.py)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_test = vectorizer.fit_transform(test_df['tweet'])
y_test = test_df['label']

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

print("✅ Evaluation complete")
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

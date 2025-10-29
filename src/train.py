import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("🟢 Starting training script...")

# Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
print("✅ Data loaded successfully")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Columns:", train_df.columns.tolist())

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['tweet'])
X_test = vectorizer.transform(test_df['tweet'])
print("✅ TF-IDF vectorization done")

y_train = train_df['label']
y_test = test_df['label']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model training complete")

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model accuracy: {acc:.4f}")

# Save model
with open("models/logreg.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully as models/logreg.pkl")

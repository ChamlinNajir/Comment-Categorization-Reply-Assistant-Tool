import pandas as pd
import pickle # Used to save our model to a file
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the clean dataset
df = pd.read_csv('comments_clean.csv')

# Handle any missing values that might have slipped through
df['clean_comment'] = df['clean_comment'].fillna('')

# 2. Split into Input (X) and Output (y)
X = df['clean_comment']  # The text
y = df['label']          # The category (Hate, Praise, etc.)

# 3. Split into Training and Testing sets (80% train, 20% test)
# random_state=42 ensures we get the same split every time we run this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} comments...")
print(f"Testing on {len(X_test)} comments...")

# 4. Convert Text to Numbers (TF-IDF Vectorization)
vectorizer = TfidfVectorizer(max_features=5000) # Keep top 5000 most important words
X_train_vec = vectorizer.fit_transform(X_train) # Learn vocabulary from train data
X_test_vec = vectorizer.transform(X_test)       # Apply same vocabulary to test data

# 5. Initialize and Train the Model
# We use Logistic Regression (simple, fast, and great for text classification)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("Model training complete!")

# 6. Quick Evaluation
# Let's see how well it did on the test data
predictions = model.predict(X_test_vec)
print("\n--- Model Accuracy ---")
print(accuracy_score(y_test, predictions))

# 7. Save the "Brain"
# We need to save BOTH the vectorizer (the translator) and the model (the brain)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('comment_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nFiles saved: 'tfidf_vectorizer.pkl' and 'comment_classifier_model.pkl'")
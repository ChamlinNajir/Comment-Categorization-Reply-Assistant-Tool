import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- LOAD RESOURCES ---
# Load the cleaned data
df = pd.read_csv('comments_clean.csv')
df['clean_comment'] = df['clean_comment'].fillna('')

# Load the saved model and vectorizer
print("Loading model and vectorizer...")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('comment_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- RECREATE TEST SET ---
# We must use the exact same random_state=42 to get the same test data as before
X = df['clean_comment']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the test data using the loaded vectorizer
X_test_vec = vectorizer.transform(X_test)

# --- PART 1: DETAILED METRICS ---
print("\n--- detailed Classification Report ---")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# --- PART 2: VISUALIZE CONFUSION MATRIX ---
# This shows us exactly where the model gets confused
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- PART 3: LIVE TESTING LOOP ---
def predict_comment(text):
    # We must use the exact same cleaning function steps
    # (Simplified here for the loop)
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Vectorize and Predict
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    
    # Get probability (confidence score)
    probs = model.predict_proba(text_vec)
    confidence = probs.max()
    
    return prediction, confidence

print("\n" + "="*50)
print("   INTERACTIVE MODE INITIALIZED")
print("   Type a comment to test the model.")
print("   Type 'exit' to quit.")
print("="*50)

while True:
    user_input = input("\nEnter a comment: ")
    if user_input.lower() == 'exit':
        break
    
    category, conf = predict_comment(user_input)
    print(f"--> Prediction: {category.upper()}")
    print(f"--> Confidence: {conf:.2f}")
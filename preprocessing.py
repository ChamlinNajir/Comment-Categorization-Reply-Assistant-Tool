import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- DOWNLOAD NLTK RESOURCES ---
# We need to download these small dictionaries once so NLTK knows what words are
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("Download complete!")

# --- INITIALIZE TOOLS ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- THE CLEANING FUNCTION ---
def clean_text(text):
    # 1. Handle empty rows
    if not isinstance(text, str):
        return ""
    
    # 2. Lowercase everything
    text = text.lower()
    
    # 3. Remove special characters and numbers (Keep only a-z)
    # The regex [^a-z\s] means "remove anything that isn't a letter or space"
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize (Split sentence into a list of words)
    words = text.split()
    
    # 5. Remove Stopwords & Lemmatize
    # - Stopwords: Remove "the", "is", "and"
    # - Lemmatize: Convert "running" -> "run", "better" -> "good"
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # 6. Join back into a string
    return " ".join(cleaned_words)

# --- EXECUTION ---
# 1. Load the dataset
df = pd.read_csv('comment_dataset.csv')

# 2. Apply the cleaning function to the 'comment' column
print("Cleaning comments... please wait.")
df['clean_comment'] = df['comment'].apply(clean_text)

# 3. Show a comparison (Before vs After)
print("\n--- Preprocessing Comparison ---")
print(df[['comment', 'clean_comment']].head())

# 4. Save the clean data
# We save this to a new file so we don't have to clean it again later
df.to_csv('comments_clean.csv', index=False)
print("\nSuccess! Cleaned data saved to 'comments_clean.csv'")
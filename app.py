import streamlit as st
import pandas as pd
import pickle
import re

# --- 1. SETUP & LOADING ---
st.set_page_config(page_title="Comment Reply Assistant", page_icon="💬")

@st.cache_resource
def load_model():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('comment_classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vec, model

vectorizer, model = load_model()

# --- 2. REPLY TEMPLATES ---
# This dictionary maps categories to suggested replies
reply_templates = {
    "praise": "Thank you so much! We are thrilled you liked it. 😊",
    "support": "Thanks for the support! It keeps us going. 🚀",
    "emotional": "We are touched that this resonated with you. ❤️",
    "constructive criticism": "Thanks for the feedback. We'll definitely look into this for next time! 🔧",
    "hate": "We do not tolerate abusive language here. 🚫",
    "spam": "[Auto-Reply] This comment has been flagged as spam.",
    "question": "Great question! We will get back to you with an answer shortly."
}

# --- 3. HYBRID PREDICTION FUNCTION ---
def predict_category(text):
    # CLEANING (Must match training cleaning exactly)
    clean_text = text.lower()
    clean_text = re.sub(r'[^a-z\s]', '', clean_text)
    
    # KEYWORD OVERRIDE (The "Quick Fix" for small datasets)
    # If the AI is weak, these rules take over.
    if "hate" in clean_text or "stupid" in clean_text or "trash" in clean_text:
        return "hate", 1.0
    if "follow" in clean_text and "me" in clean_text:
        return "spam", 1.0

    # AI PREDICTION
    vec_text = vectorizer.transform([clean_text])
    pred = model.predict(vec_text)[0]
    prob = model.predict_proba(vec_text).max()
    
    return pred, prob

# --- 4. THE USER INTERFACE ---
st.title("💬 AI Comment Assistant")
st.write("Enter a user comment to categorize it and generate a reply.")

# Text Input
user_comment = st.text_area("User Comment", placeholder="Paste a comment here...")

if st.button("Analyze Comment"):
    if user_comment:
        # Get Prediction
        category, confidence = predict_category(user_comment)
        
        # Normalize category string for looking up templates
        cat_key = category.lower().strip()
        
        # Display Results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category")
            # Color code the output
            if cat_key in ['hate', 'spam']:
                st.error(category.upper())
            elif cat_key in ['praise', 'support']:
                st.success(category.upper())
            else:
                st.warning(category.upper())
            
            st.caption(f"Confidence Score: {confidence:.2f}")

        with col2:
            st.subheader("Suggested Reply")
            # Fetch reply, default to a generic one if category not found
            reply = reply_templates.get(cat_key, "Thanks for your comment!")
            st.info(reply)
            
    else:
        st.warning("Please enter some text first.")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("This tool uses Logistic Regression + TF-IDF to classify comments.")
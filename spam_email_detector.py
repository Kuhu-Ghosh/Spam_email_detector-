import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Features and labels
X = df['message']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ANN
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, verbose=0)

# Expanded spam keyword list
spam_keywords = [
    'lottery', 'winner', 'free', 'iphone', 'i phone', 'iph0ne', 'click here',
    'claim now', 'urgent response', 'act now', 'buy now', 'guaranteed',
    '100% free', 'limited offer', 'credit card', 'win big', 'earn money'
]

def is_nonsense(msg):
    # Very short or meaningless content
    return len(msg.strip()) < 5 or re.fullmatch(r'[a-z]+', msg.strip().lower()) is not None

# Start loop
print("\n Spam Detector is Ready!")
while True:
    user_input = input("\nEnter email/message to check for spam (or type 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break

    lowered = user_input.lower()
    vectorized = vectorizer.transform([user_input]).toarray()
    prediction = model.predict(vectorized, verbose=0)[0][0]
    matched_keywords = [kw for kw in spam_keywords if kw in lowered]

    label = "HAM"

    # New override logic
    if is_nonsense(user_input):
        label = "SPAM"
        print(" Message is likely meaningless or gibberish.")
    elif len(matched_keywords) >= 2:
        label = "SPAM"
        print(f"Multiple spam keywords found: {', '.join(matched_keywords)}")
    elif prediction >= 0.7:
        label = "SPAM"
    elif prediction < 0.3 and not matched_keywords:
        label = "HAM"
    elif prediction < 0.3 and matched_keywords:
        label = "SPAM"
        print(f"Spam keywords found despite low model confidence: {', '.join(matched_keywords)}")
    else:
        label = "UNCERTAIN"
        if matched_keywords:
            print(f" Contains: {', '.join(matched_keywords)}")

    print(f"Prediction: {label} (Confidence: {prediction:.2f})")

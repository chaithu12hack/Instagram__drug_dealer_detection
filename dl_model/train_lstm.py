# dl_model/train_lstm.py

import pandas as pd
import numpy as np
import tensorflow as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from utils.preprocess import load_and_preprocess
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess dataset
X, y = load_and_preprocess("dataset/posts.csv")

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=100)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(type(X_train), type(y_train))  # should both be <class 'numpy.ndarray'>

# Train
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

# Save model and tokenizer
# Save the model
model.save('dl_model/lstm_model.h5')
print("Model saved to dl_model/lstm_model.h5")
import pickle
with open("dl_model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ LSTM model and tokenizer saved!")
print("✅ LSTM model training completed!")
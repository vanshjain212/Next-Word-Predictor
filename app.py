import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# 1. Page Config
st.set_page_config(page_title="Neural Movie Predictor", page_icon="🎬")

@st.cache_resource
def load_assets():
    # Load your trained LSTM and the Tokenizer
    model = tf.keras.models.load_model('movie_lstm.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

# 2. UI Elements
st.title("🎬 Neural Next-Word Predictor")
st.markdown("""
This AI was trained on **200,000+ movie dialogues** using a **Stacked LSTM** and **GloVe embeddings**. 
Type a sentence below to see what it suggests next!
""")

seed_text = st.text_input("Enter your movie line:", "I want to")
max_seq_len = 50# Use the same as your training

if seed_text:
    # 3. Inference Logic
    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    predicted_probs[0], predicted_probs[1] = 0, 0 # Mask Padding and OOV
    
    top_3_idx = np.argsort(predicted_probs)[-5:][::-1]
    
    # 4. Display Suggestions
    st.subheader("Top Suggestions:")
    cols = st.columns(5)
    
    for i, idx in enumerate(top_3_idx):
        for word, index in tokenizer.word_index.items():
            if index == idx:
                confidence = predicted_probs[idx] * 100
                with cols[i]:
                    st.metric(label=f"Option {i+1}", value=word, delta=f"{confidence:.2f}%")
                break
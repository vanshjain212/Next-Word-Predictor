import streamlit as st
import tensorflow as tf
import torch
import pickle
import numpy as np
import time
import os
import math
from pathlib import Path
from transformers import pipeline

# --- 1. Page Config & Custom Styling ---
st.set_page_config(page_title="Sequence Model Benchmarking", page_icon="🔬", layout="wide")

# Resolve paths (Assumes app.py is inside a 'src/' folder)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# --- 2. Hardcoded Architecture Metrics ---
LSTM_SIZE_MB = 34.0
LSTM_VAL_LOSS = 5.35
LSTM_PPL = math.exp(LSTM_VAL_LOSS)

TRANSFORMER_SIZE_MB = 330.0
TRANSFORMER_VAL_LOSS = 3.32
TRANSFORMER_PPL = math.exp(TRANSFORMER_VAL_LOSS)

# --- 3. CRITICAL: Prevent TF from hogging all GPU VRAM ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

@st.cache_resource
def load_models():
    """Loads both architectures securely using dynamic project paths."""
    
    # 1. Verify files exist before attempting to load
    lstm_path = MODEL_DIR / "movie_lstm.h5"
    tokenizer_path = MODEL_DIR / "tokenizer.pkl"
    transformer_dir = MODEL_DIR / "transformer_weights"
    
    if not lstm_path.exists() or not tokenizer_path.exists() or not transformer_dir.exists():
        st.error(f"Missing model files! Please ensure {MODEL_DIR} contains the LSTM .h5, tokenizer.pkl, and transformer_weights folder.")
        st.stop() # Halts the Streamlit app gracefully

    # 2. LOAD LSTM
    lstm_model = tf.keras.models.load_model(str(lstm_path))
    with open(str(tokenizer_path), 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)
        
    # 3. LOAD TRANSFORMER
    device = 0 if torch.cuda.is_available() else -1
    
    transformer_pipe = pipeline(
        "text-generation",
        model=str(transformer_dir),
        tokenizer=str(transformer_dir),
        device=device
    )
    
    return lstm_model, lstm_tokenizer, transformer_pipe

# Keep the rest of your UI code the exact same!
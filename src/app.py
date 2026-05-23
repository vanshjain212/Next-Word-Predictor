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

# 1. Page Config & Custom Styling
st.set_page_config(page_title="Sequence Model Benchmarking", page_icon="🔬", layout="wide")

# Resolve paths dynamically relative to this script's location
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# 2. Hardcoded Architecture Metrics (Derived from your Training Logs)
LSTM_SIZE_MB = 34.0
LSTM_VAL_LOSS = 5.35
LSTM_PPL = math.exp(LSTM_VAL_LOSS)

TRANSFORMER_SIZE_MB = 330.0
TRANSFORMER_VAL_LOSS = 3.32
TRANSFORMER_PPL = math.exp(TRANSFORMER_VAL_LOSS)

@st.cache_resource
def load_models():
    """Loads both architectures securely using dynamic project paths."""
    # --- LOAD LSTM ---
    lstm_path = str(MODEL_DIR / "movie_lstm.h5")
    tokenizer_path = str(MODEL_DIR / "tokenizer.pkl")
    
    lstm_model = tf.keras.models.load_model(lstm_path)
    with open(tokenizer_path, 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)
        
    # --- LOAD TRANSFORMER ---
    # Points to where your transformer pipeline artifact directory is stored
    transformer_dir = str(MODEL_DIR / "transformer_weights")
    device = 0 if torch.cuda.is_available() else -1
    
    transformer_pipe = pipeline(
        "text-generation",
        model=transformer_dir,
        tokenizer=transformer_dir,
        device=device
    )
    
    return lstm_model, lstm_tokenizer, transformer_pipe

# Keep the remaining part of your file (Evaluation Functions & Main UI layout) exactly as you wrote it!
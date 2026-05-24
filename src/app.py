import streamlit as st
import tensorflow as tf
import torch
import pickle
import numpy as np
import time
import math
from pathlib import Path
from transformers import pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Keras 3 Bug Fix Interceptor ---
class SafeEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        # Silently remove the bugged configuration before Keras crashes
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
        
# --- 1. Page Config & Custom Styling ---
st.set_page_config(page_title="Sequence Model Benchmarking", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    .metric-row { display: flex; justify-content: space-between; align-items: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. Dynamic Paths & VRAM Protection ---
BASE_DIR = Path(__file__).resolve().parent.parent # Assumes app.py is in src/
MODEL_DIR = BASE_DIR / "models"

# Prevent TF from hogging all GPU VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 3. Hardcoded Architecture Metrics ---
LSTM_SIZE_MB = 34.0
LSTM_VAL_LOSS = 5.35
LSTM_PPL = math.exp(LSTM_VAL_LOSS)

TRANSFORMER_SIZE_MB = 330.0
TRANSFORMER_VAL_LOSS = 3.32
TRANSFORMER_PPL = math.exp(TRANSFORMER_VAL_LOSS)

# --- 4. Model Loading ---
@st.cache_resource
def load_models():
    """Loads both architectures securely using dynamic paths."""
    # LSTM Paths
    lstm_path = MODEL_DIR / "movie_lstm.h5"
    tokenizer_path = MODEL_DIR / "tokenizer.pkl"
    
    # Transformer Path
    transformer_dir = MODEL_DIR / "transformer_weights"
    
    if not lstm_path.exists() or not transformer_dir.exists():
        st.error(f"Missing models! Please run `dvc pull` to download weights to {MODEL_DIR}")
        st.stop()

    lstm_model = tf.keras.models.load_model(
        str(lstm_path), 
        custom_objects={'Embedding': SafeEmbedding}
    )
    with open(str(tokenizer_path), 'rb') as handle:
        lstm_tokenizer = pickle.load(handle)
        
    device = 0 if torch.cuda.is_available() else -1
    transformer_pipe = pipeline(
        "text-generation",
        model=str(transformer_dir),
        tokenizer=str(transformer_dir),
        device=device
    )
    
    return lstm_model, lstm_tokenizer, transformer_pipe

with st.spinner("Initializing Neural Architectures (This might take a moment)..."):
    lstm_model, lstm_tokenizer, transformer_pipe = load_models()

# --- 5. Evaluation Functions ---
def get_diversity_score(text):
    words = text.lower().split()
    if not words: return 0
    return len(set(words)) / len(words)

def generate_lstm_paragraph(seed, model, tokenizer, length=40):
    output_text = seed
    for _ in range(length):
        token_list = tokenizer.texts_to_sequences([output_text.lower()])[0]
        token_list = pad_sequences([token_list], maxlen=49, padding='pre')
        probs = model.predict(token_list, verbose=0)[0]
        
        # Mask Padding (0) and OOV (1)
        probs[0], probs[1] = 0, 0 
        probs = probs / np.sum(probs)
        
        idx = np.random.choice(len(probs), p=probs)
        word = next((w for w, index in tokenizer.word_index.items() if index == idx), "")
        output_text += " " + word
    return output_text

# --- 6. Sidebar Configuration ---
st.sidebar.title("⚙️ Generation Parameters")
st.sidebar.markdown("Fine-tune the Transformer's output dynamically.")

generate_tokens = st.sidebar.slider("Tokens to Generate", 10, 150, 40, 10)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.1, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05)
rep_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1)

# --- 7. Main UI Layout ---
st.title("🔬 Sequence Model Benchmarking")
st.markdown("Comparing **Recurrence** (Stacked LSTM) against **Self-Attention** (GPT-2) for long-form narrative generation.")

# Interactive Tabs
tab1, tab2 = st.tabs(["🚀 Live Head-to-Head Benchmark", "📊 Architecture Deep Dive"])

with tab1:
    st.markdown("### Generate Text")
    seed_text = st.text_input("Enter a prompt to start the generation:", "It was a dark night and I was walking alone when")
    
    if st.button("Run Benchmark Analysis ⚡", use_container_width=True):
        st.divider()
        col1, col2 = st.columns(2)

        # --- LSTM COLUMN ---
        with col1:
            st.header("🧱 Stacked LSTM")
            st.caption("Sequential Processing | Hidden State Memory")
            
            start_time = time.time()
            with st.spinner("LSTM is predicting token-by-token..."):
                lstm_result = generate_lstm_paragraph(seed_text, lstm_model, lstm_tokenizer, length=generate_tokens)
            duration = time.time() - start_time
            
            throughput = generate_tokens / duration
            diversity = get_diversity_score(lstm_result)

            st.info(f"**Output:**\n\n{lstm_result}")
            
            st.subheader("Live Performance")
            im1, im2, im3 = st.columns(3)
            im1.metric("Latency", f"{duration:.2f} s")
            im2.metric("Throughput", f"{throughput:.1f} t/s")
            im3.metric("Diversity", f"{diversity:.2f}")

        # --- TRANSFORMER COLUMN ---
        with col2:
            st.header("🤖 Fine-Tuned GPT-2")
            st.caption("Parallel Processing | Multi-Head Self-Attention")
            
            start_time = time.time()
            with st.spinner("Transformer is projecting attention..."):
                trans_result = transformer_pipe(
                    seed_text, 
                    max_new_tokens=generate_tokens, 
                    do_sample=True, 
                    temperature=temperature, 
                    top_p=top_p, 
                    repetition_penalty=rep_penalty
                )[0]['generated_text']
            duration = time.time() - start_time
            
            throughput = generate_tokens / duration
            diversity = get_diversity_score(trans_result)

            st.success(f"**Output:**\n\n{trans_result}")
            
            st.subheader("Live Performance")
            tim1, tim2, tm3 = st.columns(3)
            tim1.metric("Latency", f"{duration:.2f} s")
            tim2.metric("Throughput", f"{throughput:.1f} t/s")
            tm3.metric("Diversity", f"{diversity:.2f}")

with tab2:
    st.markdown("### Under the Hood")
    st.write("This section compares the static, structural metrics of the two models resulting from the MLOps training pipeline.")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.subheader("LSTM Baseline")
        st.metric("Model Size", f"{LSTM_SIZE_MB} MB")
        st.metric("Validation Loss", f"{LSTM_VAL_LOSS}")
        st.metric("Perplexity (PPL)", f"{LSTM_PPL:.2f}")
    
    with col_stat2:
        st.subheader("Transformer Final")
        st.metric("Model Size", f"{TRANSFORMER_SIZE_MB} MB", delta=f"+{TRANSFORMER_SIZE_MB - LSTM_SIZE_MB} MB", delta_color="inverse")
        st.metric("Validation Loss", f"{TRANSFORMER_VAL_LOSS}", delta=f"{TRANSFORMER_VAL_LOSS - LSTM_VAL_LOSS:.2f}", delta_color="inverse")
        st.metric("Perplexity (PPL)", f"{TRANSFORMER_PPL:.2f}", delta=f"{TRANSFORMER_PPL - LSTM_PPL:.2f}", delta_color="inverse")
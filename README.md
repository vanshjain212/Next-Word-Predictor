# 🎬 Neural Next-Word Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Status](https://img.shields.io/badge/Status-Completed-green)

A deep learning-based text suggestion engine that predicts the next word in a sequence using a **Stacked LSTM** architecture and **GloVe Embeddings**. Trained on the **Cornell Movie-Dialogs Corpus**, this application mimics the predictive text capabilities of modern smartphone keyboards, serving probabilistic suggestions in real-time.

---

## 🚀 Live Demo

https://github.com/user-attachments/assets/8aac6783-3399-4f90-a7f1-8ca5e3faa10b

 

**Sample Predictions from the model:**

**1. Contextual Understanding (Family & Relationships)**
*Input: "I want to have a drink with my"*
![Prediction Example 1](<img width="512" height="243" alt="unnamed" src="https://github.com/user-attachments/assets/2f8c4b8f-6f0e-41e5-8fa9-360f2f79c961" />)
*The model correctly prioritizes "father", "friends", and "brother" over random nouns, showing it understands social context.*

**2. Learned Idioms (Long-term Dependencies)**
*Input: "...with my friends in a long"*
![Prediction Example 2](<img width="512" height="290" alt="unnamed" src="https://github.com/user-attachments/assets/cc5796c8-9d46-49c3-82f9-732e4dfe8aa9" />)
*The model locks onto the phrase "in a long time" with 50% confidence, demonstrating LSTM memory retention.*

---

## 🧠 Model Architecture & Technical Approach

### 1. The Core Brain: Stacked LSTM
Instead of a simple RNN (which suffers from vanishing gradients) or a single-layer LSTM, I implemented a **Stacked LSTM** architecture:
- **Layer 1:** 256 Units (Captures lower-level syntactic features like grammar).
- **Layer 2:** 128 Units (Captures higher-level semantic features like tone and intent).
- **Dropout (0.2):** Applied to prevent overfitting and force the model to learn robust generalizations.

### 2. Transfer Learning with GloVe
Rather than training embeddings from scratch (which requires massive data), I integrated **100-dimensional GloVe (Global Vectors for Word Representation)**.
- This provided the model with a pre-built "semantic map" of 15,000 words before training even began.
- Result: Faster convergence and better understanding of synonyms.

### 3. Probabilistic Inference Engine
The model doesn't just guess; it calculates the probability distribution over the entire vocabulary.
- **Softmax Layer:** Converts raw logits into probabilities summing to 100%.
- **OOV Masking:** Explicitly masks the `<OOV>` (Out-Of-Vocabulary) token during inference to force the model to predict the nearest known synonym rather than a generic "unknown" placeholder.

---

## ⚙️ Strategic Engineering Decisions

### Why 15,000 Words? (Vocabulary Optimization)
The full dataset contains 55,000+ words. I truncated this to the top **15,000 most frequent words**.
- **Reasoning:** The "long tail" of vocabulary consists of rare names and typos that introduce noise.
- **Impact:** Covered **92% of the corpus** while reducing the Softmax computation load by **72%**, allowing for real-time inference on standard hardware.

### Why 120,000 Samples? (Data Efficiency)
I trained on a strategic subset of 120,000 dialogue pairs rather than the full 300k.
- **Reasoning:** To balance model convergence with hardware RAM constraints (T4 GPU / 12GB RAM).
- **Result:** Achieved **15% Top-1 Accuracy** (high for an open-ended vocabulary task) without hitting memory bottlenecks.

---

## 💻 Tech Stack
* **Deep Learning:** TensorFlow, Keras (LSTM, Embedding, Dense)
* **NLP:** NLTK, GloVe Embeddings, Tokenizer
* **Web Framework:** Streamlit (Custom Debounced Input for real-time inference)
* **Visualization:** Matplotlib (Accuracy/Loss Curves)

---

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vanshjain212/Next-Word-Prediction.git](https://github.com/vanshjain212/Next-Word-Prediction.git)
   cd Next-Word-Prediction

Install dependencies:
Bash
pip install -r requirements.txt

Run the Streamlit App:
Bash
streamlit run app.py

🔮 Future Scope
MLOps Integration: Implementing DVC for data versioning and MLflow for experiment tracking.
Transformer Upgrade: Experimenting with a distilled GPT-2 model to compare perplexity scores against the current LSTM baseline.

Author
Vansh V Jain
B.Tech Computer Science Student | Deep Learning Enthusiast
https://www.linkedin.com/in/vansh-jain-76b66a287/ | vanjainlko@gmail.com

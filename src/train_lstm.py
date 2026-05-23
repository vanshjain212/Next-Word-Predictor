import os
import yaml
import pickle
import mlflow
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import dagshub

# 1. Initialize DagsHub/MLflow
dagshub.init(repo_owner='vanshjain212', repo_name='Next-Word-Predictor', mlflow=True)

# 2. Load Params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
p = params["train_lstm"]

# 3. Load Data & Slice for LSTM
df = pd.read_csv("data/clean_dialogues.csv")

# Only take the top N rows for the LSTM to prevent hardware crashes
df = df.head(p["subset_size"]) 
data_subset = df["text"].dropna().astype(str).tolist()
print(f"LSTM training on {len(data_subset)} rows.")

with mlflow.start_run(run_name="LSTM_Training"):
    # Log params to MLflow
    mlflow.log_params(p)

    # 4. Tokenization [cite: 798]
    tok = Tokenizer(num_words=p["vocab_size"], oov_token="<OOV>")
    tok.fit_on_texts(data_subset)
    seqs = tok.texts_to_sequences(data_subset)

    # 5. Create Sequences [cite: 799, 801]
    X, y = [], []
    for line in seqs:
        for i in range(1, len(line)):
            X.append(line[:i])
            y.append(line[i])

    X_final = np.array(pad_sequences(X, maxlen=p["max_seq_len"], padding='pre'))
    y_final = np.array(y)

    # 6. GloVe Embeddings [cite: 792, 802]
    embeddings_index = {}
    with open(p["glove_path"], encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word, coefs = values[0], np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((p["vocab_size"], p["embedding_dim"]))
    for word, i in tok.word_index.items():
        if i < p["vocab_size"]:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # 7. Model Building [cite: 809]
    model = Sequential([
        Input(shape=(p["max_seq_len"],)),
        Embedding(input_dim=p["vocab_size"], output_dim=p["embedding_dim"], 
                  weights=[embedding_matrix], trainable=False),
        LSTM(p["lstm_units_1"], return_sequences=True),
        Dropout(0.3),
        LSTM(p["lstm_units_2"]),
        Dropout(0.2),
        Dense(p["vocab_size"], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 8. Callbacks & Training [cite: 832, 834]
    early_stop = EarlyStopping(monitor='val_loss', patience=p["patience"], restore_best_weights=True)
    
    os.makedirs("models", exist_ok=True)
    history = model.fit(X_final, y_final, batch_size=p["batch_size"], epochs=p["epochs"], 
                        validation_split=0.1, callbacks=[early_stop])

    # Log metrics
    best_val_loss = min(history.history['val_loss'])
    mlflow.log_metric("val_loss", best_val_loss)
    mlflow.log_metric("perplexity", np.exp(best_val_loss))

    # 9. Save Artifacts [cite: 847, 880]
    model_path = "models/movie_lstm.h5"
    model.save(model_path)
    with open('models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Log Model Size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    mlflow.log_metric("model_size_mb", size_mb)
    print(f"LSTM Training Complete. Model Size: {size_mb:.2f} MB")
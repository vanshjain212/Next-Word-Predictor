---
title: Next Word Predictor (LSTM)
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Distributed NLP Text Generation Engine

An end-to-end Machine Learning pipeline predicting the next sequence of text using Deep Learning, containerized for cloud deployment. 

## 🏗️ Architecture & MLOps Pipeline
This project moves beyond a standard Jupyter Notebook by implementing a full, production-grade deployment architecture:

* **Deep Learning Model:** Built using **Long Short-Term Memory (LSTM)** networks and **GloVe embeddings** to understand sequential text context.
* **Dual Framework Environment:** Utilizes both **PyTorch** and **TensorFlow**, managed via carefully structured Docker layers to optimize build times and prevent RAM bottlenecks.
* **Data Version Control (DVC):** Heavy binary files (like `tokenizer.pkl` and `.h5` model weights) are strictly decoupled from Git. They are securely stored on **DagsHub** and pulled dynamically at runtime via DVC.
* **Containerization:** The entire environment is packaged into a **Docker** container to ensure absolute consistency across local and cloud environments.
* **Cloud Deployment:** Hosted on **Hugging Face Spaces** utilizing enterprise Linux infrastructure to bypass local Windows WSL2 hardware constraints.
* **User Interface:** A lightweight, interactive front-end built with **Streamlit**.

## 🚀 Try it Live
Type a starting phrase into the interface above to test the inference speed and accuracy of the model!
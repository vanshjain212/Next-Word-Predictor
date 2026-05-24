# 1. Base Image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies (Git is required for DVC)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python dependencies
# Copy the requirements file
COPY requirements.txt .

# Layer 1: Install PyTorch, save the layer, and clear RAM
RUN pip install --no-cache-dir torch==2.2.1

# Layer 2: Install TensorFlow, save the layer, and clear RAM
RUN pip install --no-cache-dir tensorflow>=2.16.0

# Layer 3: Install the rest (it skips Torch/TF since they are done)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the Hugging Face Spaces default port
EXPOSE 7860

# 7. Authenticate DVC, pull weights, and launch the app
# The variables starting with $ will be securely provided by Hugging Face Secrets
# 7. Authenticate DVC, pull weights, and launch the app
# The variables starting with $ will be securely provided by Hugging Face Secrets
CMD git init && \
    dvc remote modify origin --local auth basic && \
    dvc remote modify origin --local user $DAGSHUB_USERNAME && \
    dvc remote modify origin --local password $DAGSHUB_TOKEN && \
    dvc pull && \
    streamlit run src/app.py --server.port=7860 --server.address=0.0.0.0
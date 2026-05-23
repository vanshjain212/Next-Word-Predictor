import os
import yaml
import mlflow
import numpy as np
from datasets import Dataset
import pandas as pd
from transformers import GPT2LMHeadModel, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import dagshub

# 1. Initialize MLflow
dagshub.init(repo_owner='vanshjain212', repo_name='Next-Word-Predictor', mlflow=True)

# 2. Load Params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
p = params["train_transformer"]

# 3. Load Model & Tokenizer [cite: 885]
tokenizer = AutoTokenizer.from_pretrained(p["model_name"])
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(p["model_name"])

# 4. Prepare Dataset from the shared CSV
df = pd.read_csv("data/clean_dialogues.csv").dropna()
dataset = Dataset.from_pandas(df)
data = dataset.train_test_split(test_size=p["test_size"], seed=params["base"]["random_state"])

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128)

tokenized_datasets = data.map(tokenize_function, batched=True, remove_columns=["text", "__index_level_0__"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

with mlflow.start_run(run_name="Transformer_FineTuning"):
    mlflow.log_params(p)

    # 5. Training Setup [cite: 918, 920]
    training_args = TrainingArguments(
        output_dir="models/transformer_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=p["epochs"],
        fp16=True,
        learning_rate=float(p["learning_rate"]),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # 6. Train and Log
    train_result = trainer.train()
    
    # Calculate Perplexity from Loss
    eval_results = trainer.evaluate()
    val_loss = eval_results['eval_loss']
    perplexity = np.exp(val_loss)
    
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("perplexity", perplexity)

    # 7. Save Final Model [cite: 699]
    save_dir = "models/transformer_weights"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Transformer Training Complete. Validation Loss: {val_loss:.3f}")
# src/phase3_transformer.py
import torch
import pandas as pd
import numpy as np
from transformers import (RobertaTokenizer, RobertaForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from evaluate import load as load_metric
import os

# ── AUTO SELECT DATASET ───────────────────────────────────
if os.path.exists("../data/augmented/hc3_augmented.csv"):
    DATA_PATH = "../data/augmented/hc3_augmented.csv"
    print("✅ Using AUGMENTED dataset (2nd training — adversarial robustness)")
else:
    DATA_PATH = "../data/processed/hc3_cleaned.csv"
    print("✅ Using ORIGINAL dataset (1st training)")

# ── 1. Load & Split Data ──────────────────────────────────
df = pd.read_csv(DATA_PATH)   # ← only this line changes
train_df, test_df = train_test_split(df, test_size=0.2,
                                     stratify=df['label'], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.1,
                                     stratify=train_df['label'], random_state=42)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 2. Tokenize ───────────────────────────────────────────
MODEL_NAME = "roberta-base"
tokenizer  = RobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize(df):
    enc = tokenizer(df['text'].tolist(), truncation=True,
                    padding=True, max_length=512)
    enc['labels'] = df['label'].tolist()
    return Dataset.from_dict(enc)

train_dataset = tokenize(train_df)
val_dataset   = tokenize(val_df)
test_dataset  = tokenize(test_df)

# ── 3. Metrics ────────────────────────────────────────────
accuracy_metric = load_metric("accuracy")
f1_metric       = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1":       f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"],
    }


# ── 4. Train ──────────────────────────────────────────────
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Ensure directories exist
os.makedirs("./models/roberta_classifier", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

training_args = TrainingArguments(
    output_dir="./models/roberta_classifier",
    num_train_epochs=5,
    per_device_train_batch_size=8,        # Lowered to avoid Memory Error
    per_device_eval_batch_size=8,         # Lowered to avoid Memory Error
    gradient_accumulation_steps=2,       # Resulting effective batch size is still 16
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy="epoch",               # You already fixed this! 
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,                           # Ensure your GPU supports this (most do)
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"                     # Prevents errors if WandB/Tensorboard aren't setup
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
trainer.save_model("./models/roberta_classifier")
tokenizer.save_pretrained("./models/roberta_classifier")


# ── 5. Evaluate on Test Set ───────────────────────────────
results = trainer.evaluate(test_dataset)
print("\n=== TEST RESULTS ===")
print(f"Accuracy : {results['eval_accuracy']:.4f}")
print(f"F1 Score : {results['eval_f1']:.4f}")
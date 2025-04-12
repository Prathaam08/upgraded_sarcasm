import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, get_cosine_schedule_with_warmup
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate
import nlpaug.augmenter.word as naw

# Load dataset
df = pd.read_csv("Data/reduced_reddit_sarcasm.csv")

# Ensure required columns exist
required_columns = {"comment", "label", "situation", "who_talking_to"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Dataset must include columns: {required_columns}")

# Remove missing values
df = df.dropna(subset=["comment", "situation", "who_talking_to"])

# Balance dataset with max sample size
sample_size = min(50000, len(df))
df = df.sample(sample_size, random_state=42)

# Paraphrasing augmentation
def augment_text(text):
    aug = naw.SynonymAug(aug_src='wordnet')
    return aug.augment(text)

df["comment"] = df["comment"].apply(augment_text)

# Create combined text column
df["combined_text"] = df["comment"].astype(str) + " [SITUATION] " + df["situation"].astype(str) + " [PERSON] " + df["who_talking_to"].astype(str)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["combined_text"], df["label"], test_size=0.2, random_state=42
)

# Load tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["combined_text"], padding="max_length", truncation=True, max_length=256)

# Convert to Hugging Face dataset
train_dataset = Dataset.from_dict({"combined_text": train_texts.tolist(), "labels": train_labels.tolist()})
test_dataset = Dataset.from_dict({"combined_text": test_texts.tolist(), "labels": test_labels.tolist()})

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["combined_text"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["combined_text"])

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Load accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# # Training arguments with cosine learning rate decay
# training_args = TrainingArguments(
#     output_dir="models",
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=5,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     logging_dir="logs",
#     fp16=True,
#     warmup_steps=500,
#     learning_rate=5e-5,
#     lr_scheduler_type="cosine",
# )

training_args = TrainingArguments(
    output_dir="models",
    per_device_train_batch_size=8,  # Keep batch size small for speed
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Simulate larger batch without GPU overload
    num_train_epochs=7,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="logs",
    fp16=True,
    warmup_steps=200,  # Reduced warmup for faster convergence
    learning_rate=5e-5,
    lr_scheduler_type="cosine_with_restarts",  # Faster convergence
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Evaluate and print final accuracy
results = trainer.evaluate()
print(f"Final Accuracy: {results['eval_accuracy']:.4f}")

# Save model
trainer.save_model("models/sarcasm_detector")
tokenizer.save_pretrained("models/sarcasm_detector")

print("Model training complete and saved!")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from datasets import Dataset

nltk.download("stopwords")


def clean_text(text): 
    """Preprocess text: lowercase, remove special characters, but keep key stopwords."""
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters but keep numbers

    # Define stopwords but keep meaningful ones
    keep_words = {"not", "am", "i", "no", "who", "why", "what", "is", "was", "are", "were"}
    stop_words = set(stopwords.words("english")) - keep_words

    words = [word for word in text.split() if word not in stop_words]

    # Ensure at least one word remains
    if not words:
        return text  # Return the original text if all words were removed

    return " ".join(words)


def load_data(filepath="Data/train-balanced-sarcasm.csv", sample_size=100000):
    """Load and preprocess the dataset, reducing size to a balanced sample while preserving original ratios."""
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required_columns = {"comment", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must include columns: {required_columns}")

    # Add missing context-related columns
    if "situation" not in df.columns:
        df["situation"] = "general"
    if "who_talking_to" not in df.columns:
        df["who_talking_to"] = "unknown"

    df = df[["comment", "label", "situation", "who_talking_to"]].dropna()

    df["text"] = df["comment"].apply(clean_text)

    # Remove rows where the cleaned text is empty
    df = df[df["text"].str.strip() != ""]

    # Balance dataset while preserving original sarcasm ratio
    sarcastic = df[df["label"] == 1]
    not_sarcastic = df[df["label"] == 0]
    total_size = min(len(df), sample_size)
    sarcastic_ratio = len(sarcastic) / len(df)
    sarcastic_size = int(total_size * sarcastic_ratio)
    not_sarcastic_size = total_size - sarcastic_size

    sarcastic = sarcastic.sample(min(len(sarcastic), sarcastic_size), random_state=42)
    not_sarcastic = not_sarcastic.sample(min(len(not_sarcastic), not_sarcastic_size), random_state=42)

    # Combine & shuffle
    df = pd.concat([sarcastic, not_sarcastic]).sample(frac=1, random_state=42)

    # Apply text cleaning
    df["text"] = df["comment"].apply(clean_text)

    # Save reduced dataset
    df.to_csv("Data/reduced_reddit_sarcasm.csv", index=False)

    # Train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({
        "text": train_texts.tolist(),
        "label": train_labels.tolist(),
        "situation": df.loc[train_texts.index, "situation"].tolist(),
        "who_talking_to": df.loc[train_texts.index, "who_talking_to"].tolist()
    })

    test_dataset = Dataset.from_dict({
        "text": test_texts.tolist(),
        "label": test_labels.tolist(),
        "situation": df.loc[test_texts.index, "situation"].tolist(),
        "who_talking_to": df.loc[test_texts.index, "who_talking_to"].tolist()
    })

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = load_data()
    print("✅ Dataset reduced and saved successfully!")

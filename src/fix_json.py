# # Data/Sarcasm_Headlines_Dataset.json
# import re

# with open('Data/Sarcasm_Headlines_Dataset_v2.json', 'r') as file:
#     content = file.read()

# # Use regular expression to insert commas between adjacent JSON objects
# fixed_content = re.sub(r'}\s*{', '},{', content)

# # Optionally, wrap the content in an array if necessary
# fixed_content = f'[{fixed_content}]'

# with open('fixed_file.json', 'w') as file:
#     file.write(fixed_content)

# print("File has been fixed and saved as 'fixed_file.json'.")







# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords

# nltk.download("stopwords")
# STOPWORDS = set(stopwords.words("english"))

# def clean_text(text):
#     """Preprocess text: lowercase, remove special characters, stopwords."""
#     if not isinstance(text, str) or len(text) < 4:
#         return None  # Remove non-string or too short entries
    
#     text = text.lower()
#     text = re.sub(r"http\S+", "", text)  # Remove URLs
#     text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
#     text = " ".join([word for word in text.split() if word not in STOPWORDS])
    
#     return text if text.strip() else None  # Remove empty strings after cleaning

# def fix_json(filepath="Data/reduced_reddit_sarcasm.csv"):
#     """Fix dataset by cleaning text and removing problematic rows."""
#     df = pd.read_csv(filepath)
    
#     if "comment" not in df.columns:
#         raise ValueError("Dataset must contain a 'comment' column")
    
#     # Clean the text column
#     df["text"] = df["comment"].apply(clean_text)
    
#     # Remove rows where text is None
#     df = df.dropna(subset=["text"])
    
#     # Save the cleaned dataset
#     df.to_csv(filepath, index=False)
#     print("✅ Dataset cleaned and saved successfully!", df.isnull().sum())

# if __name__ == "__main__":
#     fix_json()





import pandas as pd
df = pd.read_csv("Data/reduced_reddit_sarcasm.csv")
print(df.isnull().sum())  # Should be all 0s
print(df[df["comment"].isna()])
print(df[df["text"].isna()]["comment"].head(10))

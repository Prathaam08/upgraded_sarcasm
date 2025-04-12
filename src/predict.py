# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# # Load model and tokenizer
# MODEL_PATH = "models/sarcasm_detector"
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# def predict_sarcasm(text, situation, who_talking_to):
#     """Predict sarcasm based on text, situation, and who the user is talking to."""
#     try:
#         combined_text = f"{text} [SITUATION] {situation} [PERSON] {who_talking_to}"
#         inputs = tokenizer(
#             combined_text, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=256  # Ensure consistency with training
#         )

#         # Move input tensors to GPU if available
#         inputs = {key: val.to(device) for key, val in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=1).item()

#         return "Sarcastic" if prediction == 1 else "Not Sarcastic"

#     except Exception as e:
#         return f"Error: {str(e)}"
# if __name__ == "__main__":
#     test_cases = [
#         ("Oh great, another meeting. Just what I needed!", "At work", "boss"),
#         ("Wow, this assignment is so much fun!", "In a classroom", "teacher"),
#         ("Nice job breaking the website, genius.", "Online comment", "unknown"),
#         ("Yeah, because waking up at 6 AM is exactly what I wanted.", "Casual conversation", "friend"),
#         ("Oh wow, another software update. I'm thrilled.", "At work", "colleague"),
#         ("Sure, I'd love to stay late and work more!", "At work", "manager"),
#         ("That was the best joke I've ever heard... not.", "Casual conversation", "friend"),
#         ("Oh yeah, this test was super easy. Totally didn’t struggle.", "In a classroom", "classmate"),
#         ("Fantastic, another pointless meeting. Just what I wanted.", "At work", "boss"),
#         ("Wow, your internet argument really changed my mind.", "Online comment", "unknown"),
#     ]

#     for text, situation, who_talking_to in test_cases:
#         result = predict_sarcasm(text, situation, who_talking_to)
#         print(f"Text: {text}\nSituation: {situation}\nWho Talking To: {who_talking_to}\nPrediction: {result}\n")



#         #1lakh train 



import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "models/sarcasm_detector"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sarcasm(text, situation, who_talking_to, threshold=0.55):
    """Predict sarcasm based on text, situation, and who the user is talking to, with confidence explanation."""
    try:
        combined_text = f"{text} [SITUATION] {situation} [PERSON] {who_talking_to}"
        inputs = tokenizer(
            combined_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            confidence = confidence.item()
            prediction = prediction.item()

        # Format confidence as a percentage
        confidence_percent = f"{confidence * 100:.2f}%"

        # Improved explanation
        if confidence < threshold:
            return f"Uncertain: The model is unsure whether this is sarcastic or not (Confidence: {confidence_percent})"
        elif prediction == 1:
            return f"Sarcastic (Confidence: {confidence_percent})"
        else:
            return f"Not Sarcastic (Confidence: {confidence_percent})"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    test_cases = [
        ("Oh great, another meeting. Just what I needed!", "At work", "boss"),
        ("Wow, this assignment is so much fun!", "In a classroom", "teacher"),
        ("Nice job breaking the website, genius.", "Online comment", "unknown"),
        ("Yeah, because waking up at 6 AM is exactly what I wanted.", "Casual conversation", "friend"),
        ("Oh wow, another software update. I'm thrilled.", "At work", "colleague"),
        ("Sure, I'd love to stay late and work more!", "At work", "manager"),
        ("That was the best joke I've ever heard... not.", "Casual conversation", "friend"),
        ("Oh yeah, this test was super easy. Totally didn’t struggle.", "In a classroom", "classmate"),
        ("Fantastic, another pointless meeting. Just what I wanted.", "At work", "boss"),
        ("Wow, your internet argument really changed my mind.", "Online comment", "unknown"),
    ]

    for text, situation, who_talking_to in test_cases:
        result = predict_sarcasm(text, situation, who_talking_to)
        print(f"Text: {text}\nSituation: {situation}\nWho Talking To: {who_talking_to}\nPrediction: {result}\n")

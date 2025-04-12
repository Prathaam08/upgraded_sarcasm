

# from flask import Flask, request, jsonify, render_template
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# import torch
# import json
# from flask import request
# import csv

# app = Flask(__name__)

# # Load model and tokenizer
# MODEL_PATH = "models/sarcasm_detector"
# tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
# model.eval()

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# def load_history():
#     try:
#         with open('history.json', 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return []  # Return empty list if no file exists
    
#     # Save history to a file after each update
# def save_history():
#     with open('history.json', 'w') as f:
#         json.dump(history, f)

# # Initialize history (load from file on startup)
# history = load_history()

# @app.route("/")
# def home():
#     return render_template("index.html")  


# @app.route("/about.html")
# def about():
#     return render_template("about.html")

# @app.route("/contact.html")
# def contact():
#     return render_template("contact.html")

# @app.route("/working.html")
# def working():
#     return render_template("working.html")



# @app.route("/predict", methods=["POST"])
# def predict():
#     global history  # Ensure history persists
#     data = request.get_json()

#     # Check for required input fields
#     if not data or "text" not in data or "situation" not in data or "who_talking_to" not in data:
#         return jsonify({"error": "Missing required fields"}), 400

#     # Construct the input text with context
#     text = f"{data['text']} [SITUATION] {data['situation']} [PERSON] {data['who_talking_to']}"
    
#     # Tokenize and move to device
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
#     inputs = {key: val.to(device) for key, val in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
#         confidence, prediction = torch.max(probs, dim=1)
#         confidence = confidence.item()
#         prediction = prediction.item()

#     # Format confidence percentage
#     confidence_percent = f"{confidence * 100:.2f}%"
#     sarcasm_label = "Sarcastic" if prediction == 1 else "Not Sarcastic"
    

#     # Apply uncertainty message for low confidence cases
#     if confidence < 0.55:
#         result = f"{sarcasm_label} (Confidence: {confidence_percent}) - The model is not sure about this sentence."
#     else:
#         result = f"{sarcasm_label} (Confidence: {confidence_percent})"

#       # Append to history
#     # Store the result in history (Appending instead of overwriting)
#     history.append({
#         "text": data["text"],
#         "situation": data["situation"],
#         "who_talking_to": data["who_talking_to"],
#         "prediction": result
#     })

#     # Save updated history to file
#     save_history()

#     return jsonify({"prediction": result})

# @app.route("/history")
# def history_page():
#     return render_template("history.html")

# @app.route("/get_history")
# def get_history():
#     return jsonify({"history": history})  # Returns full history


# import os

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     try:
#         data = request.json

#         # Ensure the models directory exists
#         feedback_path = os.path.join('Data', 'user_feedback.csv')
#         os.makedirs(os.path.dirname(feedback_path), exist_ok=True)

#         with open(feedback_path, 'a', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 data['text'],
#                 data['situation'],
#                 data['who_talking_to'],
#                 data['feedback']
#             ])
#         return "Thanks for your feedback!"
#     except Exception as e:
#         print("Error in /feedback:", e)
#         return "Internal Server Error", 500


# if __name__ == "__main__":
#     app.run(debug=True)








from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
from flask import request
import csv

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = "models/sarcasm_detector"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_history():
    try:
        with open('history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # Return empty list if no file exists
    
    # Save history to a file after each update
def save_history():
    with open('history.json', 'w') as f:
        json.dump(history, f)

# Initialize history (load from file on startup)
history = load_history()

@app.route("/")
def home():
    return render_template("index.html")  


@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/contact.html")
def contact():
    return render_template("contact.html")

@app.route("/working.html")
def working():
    return render_template("working.html")



@app.route("/predict", methods=["POST"])
def predict():
    global history  # Ensure history persists
    data = request.get_json()

    # Check for required input fields
    if not data or "text" not in data or "situation" not in data or "who_talking_to" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    # Construct the input text with context
    text = f"{data['text']} [SITUATION] {data['situation']} [PERSON] {data['who_talking_to']}"
    
    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
        confidence, prediction = torch.max(probs, dim=1)
        confidence = confidence.item()
        prediction = prediction.item()

    # Format confidence percentage
    confidence_percent = f"{confidence * 100:.2f}%"
    sarcasm_label = "Sarcastic" if prediction == 1 else "Not Sarcastic"
    

    # Apply uncertainty message for low confidence cases
    if confidence < 0.55:
        result = f"{sarcasm_label} (Confidence: {confidence_percent}) - The model is not sure about this sentence."
    else:
        result = f"{sarcasm_label} (Confidence: {confidence_percent})"

      # Append to history
    # Store the result in history (Appending instead of overwriting)
    history.append({
        "text": data["text"],
        "situation": data["situation"],
        "who_talking_to": data["who_talking_to"],
        "prediction": result
    })

    # Save updated history to file
    save_history()

    return jsonify({"prediction": result})

@app.route("/history")
def history_page():
    return render_template("history.html")

@app.route("/get_history")
def get_history():
    return jsonify({"history": history})  # Returns full history


import os

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json

        # Ensure the models directory exists
        feedback_path = os.path.join('Data', 'user_feedback.csv')
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)

        with open(feedback_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                data['text'],
                data['situation'],
                data['who_talking_to'],
                data['feedback']
            ])
        return "Thanks for your feedback!"
    except Exception as e:
        print("Error in /feedback:", e)
        return "Internal Server Error", 500


if __name__ == "__main__":
    app.run(debug=True)
This project implements a sarcasm detection model using machine learning techniques. It processes text data and predicts whether the given sentence is sarcastic or not. The application provides a web interface where users can enter text and receive predictions along with detailed explanations for the classification.

Requirements
Make sure you have the following dependencies installed:

Python 3.7+
Libraries:
fastapi
uvicorn
scikit-learn
nltk
pandas
pickle

You can install all dependencies by running:
pip install -r requirements.txt

🚀 Setup & Usage

1. Clone this repository:

2. Install Dependencies:
pip install -r requirements.txt

3. Dataset
The dataset is expected to be in the Data/reduces_reddit_sarcasm.csv file. You can get the dataset from Kaggle or another source.

4. Preprocess Data
run: python src/preprocess.py

5. Train the Model
run : python src/train.py

6. run : python app.py

The server will start

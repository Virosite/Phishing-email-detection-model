Phishing Email Detection Model

This project is an AI-based phishing email detection system built using BERT and Transformers in Python.
It classifies email text as Phishing or Legitimate using a pre-trained NLP model.

Features
Detects phishing emails based on message text
Built with BERT Transformer for natural language understanding
Simple to run locally
Lightweight and beginner-friendly

Tech Stack
Language: Python
Libraries: Transformers, Torch, NumPy, Pandas

Installation
Follow these simple steps to set up the project:

# 1. Install Python (version 3.8 or above)
# Download from https://www.python.org/downloads/
# 2. Clone this repository
git clone https://github.com/Virosite/Phishing-email-detection-model.git
cd Phishing-email-detection-model
# 3. Install required libraries
pip install torch torchvision torchaudio
pip install transformers
pip install numpy pandas

🧪 Usage
You can test the model with this simple Python script:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./model")
# Example email text
text = "Your account has been suspended. Click here to verify your details."
# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()
# Display result
if prediction == 1:
    print("🚨 Phishing Email Detected")
else:
    print("✅ Legitimate Email")

📁 Project Structure
Phishing-email-detection-model/
│
├── model/                  # Trained BERT model files
├── app.py                  # Optional demo script
├── README.md               # Project documentation
└── .gitattributes          # Git LFS tracking

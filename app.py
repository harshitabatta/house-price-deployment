from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load saved model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {hf_token}"}


# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"


def summarize(text):
    payload = {
        "inputs": text,
        "parameters": {"min_length": 30, "max_length": 60},
        "options": {"wait_for_model": True}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["summary_text"] if response.ok else "Summary failed"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expect JSON
    df = pd.DataFrame([data])

    # Predict price
    processed = preprocessor.transform(df)
    price = round(model.predict(processed)[0], 2)

    # Generate house description
    desc = f"""This is a {data['BHK']} BHK {data['Type']} located in {data['Location']} with an area of {data['Area']} sq ft. 
It is {data['Age']} years old and comes with Gym: {data['Gym']}, Lift: {data['Lift']}, and Parking: {data['Parking']}. 
The estimated price is ₹{price} Lakhs."""
    
    # Summarize
    summary = summarize(desc)

    return jsonify({
        "predicted_price": price,
        "generated_summary": summary
    })

if __name__ == "__main__":
    app.run(debug=True)

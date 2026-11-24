# House Price Prediction 🏠

A machine learning-based house price prediction application that estimates residential property prices based on property characteristics and generates AI-powered descriptions of predicted properties.

The project combines a trained machine learning model with a Flask API and Hugging Face's NLP summarization model to provide both **price predictions and human-readable property summaries**.

## ✨ Features

- **House Price Prediction:** Predict estimated property prices using a trained machine learning model.
- **Property-Based Predictions:** Accept property details such as BHK, property type, location, area, age, gym, lift, and parking availability.
- **Data Preprocessing:** Process input features using a saved preprocessing pipeline before passing them to the ML model.
- **AI-Generated Property Summaries:** Generate natural-language summaries describing the property and its estimated price.
- **REST API:** Expose prediction functionality through a Flask API endpoint.
- **JSON Responses:** Return predicted prices and generated summaries in JSON format.
- **Deployment Ready:** Configured to use environment-based port settings for deployment platforms such as Render.

## ⚙️ How It Works

The application follows a simple prediction pipeline:

1. Receive property details through a POST request.
2. Convert the input JSON into a Pandas DataFrame.
3. Transform the input using the saved preprocessor.
4. Pass the processed features to the trained machine learning model.
5. Generate an estimated house price.
6. Construct a property description using the provided details and predicted price.
7. Send the description to the Hugging Face summarization API.
8. Return the predicted price and AI-generated summary.

```text
Property Details
      │
      ▼
   Flask API
      │
      ▼
  Preprocessor
      │
      ▼
 ML Prediction Model
      │
      ▼
 Predicted Price
      │
      ▼
 Property Description
      │
      ▼
 Hugging Face NLP
      │
      ▼
 Price + AI Summary
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | REST API development |
| Scikit-learn | Machine learning model and preprocessing pipeline |
| Pandas | Input data processing |
| Joblib | Loading serialized ML model and preprocessor |
| Requests | API communication |
| Hugging Face | AI-powered text summarization |
| BART | Text summarization model |
| Render | Deployment platform |

## 📊 Input Features

The prediction API accepts the following property information:

| Feature | Description |
|---|---|
| `BHK` | Number of bedrooms, halls, and kitchens |
| `Type` | Property type |
| `Location` | Property location |
| `Area` | Property area in square feet |
| `Age` | Property age |
| `Gym` | Gym availability |
| `Lift` | Lift availability |
| `Parking` | Parking availability |

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip
- Hugging Face API token

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/house-price-prediction-ml.git
   ```

2. Navigate to the project directory:

   ```bash
   cd house-price-prediction-ml
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the trained model and preprocessor files are available in the project directory:

   ```text
   model.pkl
   preprocessor.pkl
   ```

5. Create a `.env` file in the project root:

   ```env
   HF_TOKEN=your_huggingface_api_token
   ```

6. Start the Flask application:

   ```bash
   python app.py
   ```

The application runs on port `5000` by default.

## 🔌 API Usage

### Endpoint

```http
POST /predict
```

### Request Body

```json
{
  "BHK": 2,
  "Type": "Apartment",
  "Location": "Mumbai",
  "Area": 850,
  "Age": 5,
  "Gym": "Yes",
  "Lift": "Yes",
  "Parking": "Yes"
}
```

### Example Response

```json
{
  "predicted_price": 85.5,
  "generated_summary": "A modern 2 BHK apartment in Mumbai with 850 sq ft of area..."
}
```

> The predicted price is returned in lakhs, as indicated by the application's prediction formatting.

## 🤖 AI Summarization

The application uses Hugging Face's inference API with the **`facebook/bart-large-cnn`** model to generate concise summaries of property descriptions.

The generated text incorporates:

- Property configuration
- Property type
- Location
- Area
- Age
- Amenities
- Estimated price

## 📁 Project Structure

```text
house-price-prediction-ml/
│
├── app.py
├── model.pkl
├── preprocessor.pkl
├── requirements.txt
├── .env
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Flask application and prediction API |
| `model.pkl` | Serialized machine learning model |
| `preprocessor.pkl` | Saved preprocessing pipeline |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables and API credentials |

## 🌐 Deployment

The Flask application is configured to use the `PORT` environment variable, making it suitable for deployment on platforms such as Render.

The Hugging Face API token should be configured securely through the deployment platform's environment variables.

## 🔐 Security

- Store API tokens in environment variables.
- Do not commit `.env` files or credentials to GitHub.
- Keep API credentials separate from source code.

## 📌 Project Scope

This project demonstrates the integration of:

- Machine learning-based prediction
- Data preprocessing pipelines
- Flask REST API development
- External NLP API integration
- Automated text summarization
- Model deployment

---

**Combining machine learning predictions with natural-language AI to make property price estimation more informative.** 🏡

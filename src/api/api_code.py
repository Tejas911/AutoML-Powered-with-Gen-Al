import pickle
from flask import Flask, request, jsonify
import pandas as pd

# NOTE: If your model is for regression, use the line below; otherwise, comment it.
# from pycaret.regression import load_model

# NOTE: If your model is for classification, use the line below; otherwise, comment it.
# from pycaret.classification import load_model

# Initialize the Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = r"best_model"
model = load_model(MODEL_PATH)  # Add your actual model name here.

@app.route("/")
def home():
    return "Model API is running!"

# Endpoint to make predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the data from the request
        data = request.json

        # Validate input
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input! 'features' key is required."}), 400

        # Extract features and make prediction
        features = data["features"]  # Expecting a list of dictionaries
        features_df = pd.DataFrame(features)  # Convert to DataFrame

        # Debugging logs (optional)
        print(type(features_df))
        print(features_df)

        # Make prediction
        prediction = model.predict(features_df)  # Pass DataFrame directly

        # Return the prediction
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

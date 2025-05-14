from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and label encoder
with open("health_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Feature names from your dataset
input_columns = ['Fever', 'Cough', 'Fatigue', 'Shortness of Breath', 'Blood Sugar Level', 'Blood Pressure']

# Recovery plans per disease
recovery_plans = {
    "Diabetes": {
        "health": [
            "Monitor blood sugar daily.",
            "Take insulin or medication regularly.",
            "Exercise 30 minutes daily."
        ],
        "diet": [
            "Low-sugar fruits like berries and apples.",
            "Whole grains, leafy greens, and legumes.",
            "Avoid sugary drinks and processed snacks."
        ]
    },
    "Hypertension": {
        "health": [
            "Reduce stress through meditation or yoga.",
            "Check BP daily.",
            "Limit alcohol intake."
        ],
        "diet": [
            "Eat foods rich in potassium (bananas, spinach).",
            "Avoid salty and fried foods.",
            "Drink plenty of water."
        ]
    },
    "Covid": {
        "health": [
            "Isolate and rest well.",
            "Monitor oxygen levels and temperature.",
            "Follow your doctor’s advice strictly."
        ],
        "diet": [
            "Consume immune-boosting foods (turmeric, garlic).",
            "Drink warm fluids.",
            "Stay hydrated and eat soft, nutritious meals."
        ]
    },
    "Flu": {
        "health": [
            "Wash your hands often with soap and water.",
            "Avoid touching your face, especially your eyes, nose, and mouth.",
            "Cover coughs and sneezes with a tissue or your elbow.",
        ],
        "diet": [
            "Eat a balanced diet, stay hydrated, get regular exercise, and sleep well.",
            "Avoid close contact with people who are sick.",
            "Clean and disinfect frequently touched surfaces regularly.",
        ]
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            inputs = [float(request.form[col]) for col in input_columns]
            df = pd.DataFrame([inputs], columns=input_columns)

            pred_encoded = model.predict(df)
            pred_label = label_encoder.inverse_transform(pred_encoded)[0]

            plan = recovery_plans.get(pred_label, {"health": ["No info found"], "diet": ["No info found"]})

            return render_template("result.html", prediction=pred_label, plan=plan)

        except Exception as e:
            return f"Error: {e}"

    return render_template("form.html", input_columns=input_columns)

if __name__ == "__main__":
    app.run(debug=True)

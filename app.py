from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("fraud_model.pkl")

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form input to float list
        features = [float(x) for x in request.form['features'].split(',')]

        # Validate number of features
        if len(features) != 30:
            raise ValueError("Please enter exactly 30 numeric values.")

        # Make prediction
        prediction = model.predict([features])[0]
        result = "FRAUD" if prediction == 1 else "NORMAL"
        color = "red" if prediction == 1 else "green"

        return render_template(
            'index.html',
            prediction_text=f"Prediction: {result}",
            result_color=color
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"❌ Error: {str(e)}",
            result_color="orange"
        )

if __name__ == '__main__':
    app.run(debug=True)

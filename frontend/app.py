from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../model/elnino_model.pkl"
)

model = joblib.load(MODEL_PATH)

print("✅ Model Loaded Successfully")
print("Features expected:", model.n_features_in_)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form
        oni = float(request.form["oni"])
        rainfall = float(request.form["rainfall"])
        departure = float(request.form["departure"])
        temperature = float(request.form["temperature"])

        # Create input array
        input_data = np.array([[
            oni,
            rainfall,
            departure,
            temperature
        ]])

        print("Input Data:", input_data)

        # Prediction
        prediction = model.predict(input_data)

        print("Prediction:", prediction)

        result = round(float(prediction[0]), 2)

        # Climate classification
        if oni >= 0.5:
            climate = "El Niño"
        elif oni <= -0.5:
            climate = "La Niña"
        else:
            climate = "Neutral"

        return render_template(
            "index.html",
            prediction=result,
            climate=climate
        )

    except Exception as e:

        print("ERROR:", e)

        return render_template(
            "index.html",
            prediction="Error: " + str(e),
            climate="N/A"
        )


if __name__ == "__main__":
    app.run(debug=True)
    

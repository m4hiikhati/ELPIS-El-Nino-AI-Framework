from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)


# Load trained model
model = joblib.load("../model/elnino_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Getting values from frontend
        oni = float(request.form["oni"])
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])


        # Creating input array
        input_data = np.array(
            [[oni, rainfall, temperature, humidity]]
        )


        # Prediction
        prediction = model.predict(input_data)


        result = prediction[0]


        return render_template(
            "index.html",
            prediction=result
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error : " + str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
    
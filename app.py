from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.save", "rb"))
scaler = pickle.load(open("transform.save", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features = np.array([input_features])
    
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    return render_template("home.html", prediction_text="Predicted Temperature: {}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)

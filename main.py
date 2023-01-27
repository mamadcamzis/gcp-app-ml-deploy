from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)
model_pk = pickle.load(open("flower-v1.pkl", "rb"))

@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return "Please send Post request"
    elif request.method == "POST":
        data = request.get_json()

        sepal_length = float(data["SepalLengthCm"])
        sepal_width = float(data["SepalWidthCm"])
        petal_length = float(data["PetalLengthCm"])
        petal_width = float(data["PetalWidthCm"])

        in1 = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        predictions = model_pk.predict(in1)
        #print(f"Predictions {predictions}")

        return f"Model Prediction {predictions}"
   
    
if __name__ == '__main__':
    
    app.run()


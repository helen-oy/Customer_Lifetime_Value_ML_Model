# Creating a flask app
import joblib
from flask import Flask, request, jsonify
from google.cloud import aiplatform


app = Flask(__name__)

# Loading model locally to connect to endpoint
endpoint = aiplatform.Endpoint(endpoint_name = "projects/CLV-Churn-Prediction/locations/us-central1/endpoints/clv-churn-endpoint")

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    prediction =  endpoint.predict(instances =[data])
    return jsonify(prediction.predictions[0])

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)


print(Flask.__version__)
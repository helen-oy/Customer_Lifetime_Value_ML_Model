# Creating a flask app
import joblib
from flask import Flask, request, jsonify, render_template
from google.cloud import aiplatform


app = Flask(__name__)

# Loading model locally to connect to endpoint
endpoint = aiplatform.Endpoint(endpoint_name = "projects/clv-churn-prediction/locations/us-central1/endpoints/clv-churn-endpoint")

@app.route('/')
def home():
    return render_template('home.html')  # Your input form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        recency = int(request.form['Recency'])
        frequency = int(request.form['Frequency'])
        monetary = float(request.form['Monetary'])

        # Format data for Vertex AI
        data = {
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary
        }

        # Get prediction
        prediction = endpoint.predict(instances=[data])
        ltv_cluster = prediction.predictions[0]

        return render_template('result.html', 
                             cluster=ltv_cluster,
                             recency=recency,
                             frequency=frequency,
                             monetary=monetary)

    except KeyError as e:
        return f"Missing field: {str(e)}", 400
    except ValueError as e:
        return f"Invalid input: {str(e)}", 400
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
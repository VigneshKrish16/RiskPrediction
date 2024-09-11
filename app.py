import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from risk import ImprovedAllDiseaseRiskPredictionModel, load_data

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'disease_risk_model.joblib'

def initialize_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        return ImprovedAllDiseaseRiskPredictionModel.load_model(MODEL_PATH)
    else:
        print("Training new model...")
        model = ImprovedAllDiseaseRiskPredictionModel(n_splits=5)
        file_path = "dataset/data.csv"
        data = load_data(file_path)
        X_processed, y = model.preprocess_data(data)
        model.build_models()
        model.train_and_evaluate_models(X_processed, y)
        model.save_model(MODEL_PATH)
        return model

global_model = initialize_model()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    try:
        input_data = request.get_json()
        new_patient_data = pd.DataFrame(input_data, index=[0])
        new_patient_processed = global_model.preprocess_data(new_patient_data, is_training=False)
        risk_predictions = global_model.predict_risks(new_patient_processed)

        formatted_predictions = {}
        for disease, prediction in risk_predictions.items():
            probability = prediction['probability'][0]
            risk_level = prediction['risk_level'][0]
            formatted_predictions[disease] = {
                'probability': float(probability),
                'risk_level': risk_level
            }

        return jsonify(formatted_predictions)

    except KeyError as e:
        return jsonify({'error': f'Missing key in input data: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value in input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
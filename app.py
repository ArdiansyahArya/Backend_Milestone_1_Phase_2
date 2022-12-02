import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# App Initialization
app = Flask(__name__)

# Load The Models
with open('preprocess_pipeline.pkl', 'rb') as file_1:
  preprocess_pipeline = joblib.load(file_1)

from tensorflow.keras.models import load_model
model_ann = load_model('churn_model.h5')

# Route : HomepOnlineBackup
@app.route('/')
def home():
    return '<h1> Backend Berjalan! </h1>'

@app.route('/predict', methods=['POST'])
def titanic_predict():
    args = request.json

    data_inf = {
        'tenure': args.get('tenure'),
        'MonthlyCharges': args.get('MonthlCharges'),
        'TotalCharges': args.get('TotalCharges'),
        'Dependents': args.get('Dependents'),
        'OnlineSecurity': args.get('OnlineSecurity'),
        'OnlineBackup': args.get('OnlineBackup'),
        'DeviceProtection': args.get('DeviceProtection'),
        'TechSupport': args.get('TechSupport'),
        'Contract': args.get('Contract'),
        'PaperlessBilling': args.get('PaperlessBilling'),
        'SeniorCitizen': args.get('SeniorCitizen'),
    }

    print('[DEBUG] Data Inference : ', data_inf)
    
    # Transform Inference-Set
    data_inf = pd.DataFrame([data_inf])
    data_inf_transform = preprocess_pipeline.transform(data_inf)
    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

    if y_pred_inf == 0:
        label = 'Customer most likely will not churn'
    else:
        label = 'Customer most likely will churn'

    print('[DEBUG] Result : ', y_pred_inf, label)
    print('')

    response = jsonify(
        result = str(y_pred_inf),
        label_names = label
    )

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
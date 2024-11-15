from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


model = joblib.load('server/final_model.pkl')

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates', 'alcohol'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

   
    features = [
        float(data['fixedAcidity']),
        float(data['volatileAcidity']),
        float(data['citricAcid']),
        float(data['residualSugar']),
        float(data['chlorides']),
        float(data['freeSulfurDioxide']),
        float(data['totalSulfurDioxide']),
        float(data['density']),
        float(data['pH']),
        float(data['sulphates']),
        float(data['alcohol'])
    ]

    
    features_df = pd.DataFrame([features], columns=feature_names)

  
    prediction = model.predict(features_df)
    
   
    return jsonify({'quality': str(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

MODEL_FILE = 'rf_model.pkl'

def load_model():
    with open(MODEL_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']

model, features = None, None
try:
    model, features = load_model()
except Exception as e:
    print('Model not found. Run train_model.py first to create rf_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Run training first.'}), 500
    data = request.json
    # expected JSON keys equal to features
    try:
        X = [data[f] for f in features]
    except Exception as e:
        return jsonify({'error': 'Invalid input. Expected features: ' + ','.join(features)}), 400
    arr = np.array(X).reshape(1, -1)
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr).max()
    return jsonify({'prediction': int(pred), 'confidence': float(prob)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

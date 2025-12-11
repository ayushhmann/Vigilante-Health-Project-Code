Files:
- preprocess.py : helper functions for preprocessing
- train_model.py : generates synthetic data, trains RandomForest, saves model (rf_model.pkl) and sample csv
- app.py : Flask API to load saved model and predict risk labels
- sample_input.json : example JSON for API test
- requirements.txt : python deps

Usage:
1. Create a virtualenv and install:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt

2. Train the model:
   python train_model.py
   -> This creates rf_model.pkl and synthetic_vigilante_data.csv

3. Run the API:
   python app.py

4. Test prediction:
   curl -X POST http://127.0.0.1:5000/predict -H 'Content-Type: application/json' -d @sample_input.json


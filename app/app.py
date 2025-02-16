# app.py
from flask import Flask, render_template, request
import pickle
import os
import numpy as np
from linear_regression import Lasso, Ridge, Normal, LassoPenalty, RidgePenalty, NoRegularization

app = Flask(__name__, template_folder='templates')

# Load Existing Model from .model File
# Load the trained model
model = pickle.load(open('models/A2_predicting_car_price.model','rb'))

# Verify 'theta' Attribute Immediately After Loading
if hasattr(model, 'theta'):
    print("Theta found from loaded model:", model.theta)
else:
    print("Theta not found from loaded model.")

# Load Scaler
scaler = pickle.load(open('models/scaler.pkl','rb'))

# Brand Encoding and Transmission Mapping
brand_encoded_map = {
    'Ambassador': 20, 'Ashok': 27, 'Audi': 10, 'BMW': 11, 'Chevrolet': 29,
    'Daewoo': 9, 'Datsun': 26, 'Fiat': 19, 'Force': 28, 'Ford': 4,
    'Honda': 7, 'Hyundai': 6, 'Isuzu': 14, 'Jaguar': 21, 'Jeep': 22,
    'Kia': 2, 'Land': 30, 'Lexus': 8, 'MG': 0, 'Mahindra': 1,
    'Maruti': 12, 'Mercedes-Benz': 24, 'Mitsubishi': 15, 'Nissan': 5,
    'Opel': 16, 'Peugot': 3, 'Renault': 13, 'Skoda': 18, 'Tata': 23,
    'Toyota': 17, 'Volkswagen': 25, 'Volvo': 31
}
transmission_mapping = {'Manual': 1, 'Automatic': 0}

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        brand = request.form['brand'].strip()
        year = int(request.form['year'])
        transmission = request.form['transmission'].strip()
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])

        if brand not in brand_encoded_map:
            return render_template('result.html', error=f"Brand '{brand}' not recognized.")
        if transmission not in transmission_mapping:
            return render_template('result.html', error="Invalid transmission type.")

        brand_encoded = brand_encoded_map[brand]
        transmission_encoded = transmission_mapping[transmission]
        # Prepare the input features
        features = np.array([[brand_encoded, engine, max_power, transmission_encoded, year]])
        print(f"Features before scaling: {features}")

        # Scale the features
        features_scaled = scaler.transform(features)
        print(f"Features after scaling: {features_scaled}")

        features_scaled = np.insert(features_scaled, 0, 1, axis=1) 
        # Predict the log-transformed price
        predicted_log_price = model.predict(features_scaled)
        print(f"Predicted log price: {predicted_log_price}")

        # Convert to original scale
        predicted_price = float(np.exp(predicted_log_price.flatten()[0]))
        print(f"Predicted price: {predicted_price}")

        #features = np.array([[brand_encoded, year, transmission_encoded, engine, max_power]])
        #features_scaled = scaler.transform(features)

        #print(f"Model Weights (theta): {getattr(model, 'theta', 'No theta found')}")
        #print(f"Features Scaled: {features_scaled}")

        #predicted_log_price = model.predict(features_scaled)

        #if predicted_log_price.size == 0 or np.isnan(predicted_log_price[0]):
        #    predicted_price = 0.0
        #else:
        #    predicted_price = float(np.exp(predicted_log_price.flatten()[0]))

        #print(f"Predicted Log Price: {predicted_log_price}")
        #print(f"Predicted Price (after exp): {predicted_price}")

        return render_template(
            'result.html',
            brand=brand,
            year=year,
            transmission=transmission,
            engine=engine,
            max_power=max_power,
            predicted_price=round(predicted_price, 2)
        )

    except Exception as e:
        return render_template('result.html', error=str(e))

# Run Flask App
if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5001))
    app.run(host=host, port=port)

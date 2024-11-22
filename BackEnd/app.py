from flask import Flask, request, jsonify
from keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

model = load_model('house_price_prediction_model.h5')
area_encoder = joblib.load('area_encoder.pkl')
zipcode_encoder = joblib.load('zipcode_encoder.pkl')

scaler_x = joblib.load('feature_scale.pkl')
scaler_y = joblib.load('target_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        city = data['city']
        area = data['area']
        zipcode = data['zipcode']
        stories = data['stories']
        beds = data['beds']
        baths = data['baths']
        lot_size = data['lot_size']
        year_built = data['year_built']


	    area_encoded = area_encoder.transform([area])[0]
	    zipcode_encoded = zipcode_encoder.transform([zipcode])[0]

	    features = np.array([[area_encoded, zipcode_encoded, stories, beds, baths, lot_size, year_built]])	

        features_scaled = scaler_x.transform(features)

	    features_scaled = features_scaled.reshape((1, features_scaled.shape[1]))


        predicted_price_scaled = model.predict(features_scaled)

	    predicted_price = scaler_y.inverse_transform(predicted_price_scaled)


        return jsonify({'predicted_price': float(predicted_price[0][0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


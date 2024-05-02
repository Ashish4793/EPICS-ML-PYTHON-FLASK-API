from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the machine learning model
with open('svm_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_features = request.json['inputData']['input_features']
    predicted_label = model.predict([input_features])[0]
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app

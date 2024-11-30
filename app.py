from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

# Load the trained model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and convert to a DataFrame
        data = {key: [float(value)] for key, value in request.form.items()}
        input_df = pd.DataFrame(data)

        # Ensure the input DataFrame has the same columns as the training data
        expected_features = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
            'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
            'cons.idx', 'period'
        ]  # Add all feature names used during training except 'y'
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0  # or some default value

        # Make prediction
        prediction = model.predict(input_df)

        # Return result as JSON
        return jsonify(prediction_text='Predicted Class: {}'.format(prediction[0]))
    except ValueError as e:
        print(e)
        return jsonify(prediction_text='Error: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)
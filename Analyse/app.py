from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('modele_adaboost.pkl')

df = pd.read_csv('../spotify.csv', encoding='latin1')

X = df[['danceability_%', 'valence_%', 'energy_%',
        'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    input_data = {feature: float(request.form.get(feature, 0.0)) for feature in X.columns}
    print("Données d'entrée reçues:", input_data)
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_array)
    return render_template('prediction.html', prediction=prediction[0])



if __name__ == '__main__':
    app.run(debug=True)


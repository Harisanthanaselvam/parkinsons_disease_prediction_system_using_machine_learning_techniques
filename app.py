from flask import Flask, render_template, request
import pickle
import numpy as np
from pymongo import MongoClient
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


client = MongoClient('mongodb://localhost:27017/park')

try:
 
    client.admin.command('ping')
    logger.info("MongoDB connection successful!")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")

db = client['park']
collection = db['predictions']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        jitter = float(request.form['jitter'])
        shimmer = float(request.form['shimmer'])
        nhr = float(request.form['nhr'])
        hnr = float(request.form['hnr'])
        rpde = float(request.form['rpde'])

       
        features = np.array([[jitter, shimmer, nhr, hnr, rpde]])

    
        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Parkinson's Disease Detected"
            color = "red"
        else:
            result = "No Parkinson's Disease"
            color = "green"

        record = {
            'jitter': jitter,
            'shimmer': shimmer,
            'nhr': nhr,
            'hnr': hnr,
            'rpde': rpde,
            'prediction': int(prediction),
            'result': result
        }
        collection.insert_one(record)
        logger.info("Prediction record saved to MongoDB.")

       
        sample_inputs = {
            'jitter': jitter,
            'shimmer': shimmer,
            'nhr': nhr,
            'hnr': hnr,
            'rpde': rpde
        }

        return render_template('index.html', result=result, color=color, sample=sample_inputs)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)

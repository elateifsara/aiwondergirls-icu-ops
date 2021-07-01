import numpy as np
import pandas as pd 
import json
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import requests
from math import inf

app = Flask(__name__) #Initialize the flask App
app.config['UPLOAD_FOLDER'] = "./files"

def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return filename
    else:
        return None

def get_pred(JSON_DATA=None):
    """
    url = "http://test-wids-widsdb2.default-tenant.app.mlops1.iguazio-c0.com/"
    headers = {
                "Content-Type": "application/json",
                "X-v3io-function": "widsdb2-lightgbm-serving",
                "Authorization": "Basic YXJ1bmFfbGFua2E6cXovQ09gcmA0QiFK"
            }
    payload = JSON_DATA #the ones mentioned in --data-raw and that has inputs
    try:
        response = requests.post(url, json=payload, headers=headers)
        js_resp = response.text
        output = js_resp["outputs"]
    """
    # DUMMY COMMENT WHEN API WORKS AGAIN
    output = [0.1234]
    return output
          
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        print("Upload File works")
        df = pd.read_csv("./files/"+filename)
        JSON_DATA = {"inputs":df.values.tolist()}
        """
        url = "http://test-wids-widsdb2.default-tenant.app.mlops1.iguazio-c0.com/"
        headers = {
                    "Content-Type": "application/json",
                    "X-v3io-function": "widsdb2-lightgbm-serving",
                    "Authorization": "Basic YXJ1bmFfbGFua2E6cXovQ09gcmA0QiFK"
                }
        payload = JSON_DATA #the ones mentioned in --data-raw and that has inputs
        try:
            response = requests.post(url, json=payload, headers=headers)
            js_resp = response.text
            output = js_resp["outputs"]
        """
        output = [0.1234]
        predictions = output     
        print("API WORKS")   
        return render_template('index.html', prediction_text='Predictions $ {}'.format(predictions))


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(debug=True)

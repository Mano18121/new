# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:04:15 2022

@author: karthikeyan
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('frontpage.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    
    -------
    None.
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction=model.predict(final_features)
   
    
    output = prediction[0]
    text= "The Recommended Crop is "
    return render_template("FRONTPAGE.html",output=text+output)
    #return render_template('index.html',prediction_text='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)

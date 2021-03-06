from flask import Flask, render_template, request
import jsonify
import requests
from joblib import dump, load
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = load('predition.joblib')
@app.route('/',methods=['GET'])
def Home():
    return render_template('video.html')


@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        crim = float(request.form['crim'])
        zn = float(request.form['zn'])
        indus = float(request.form['indus'])
        chas = request.form['chas']
        if (chas == '1'):
            chas_detail = 1
        else:
            chas_detail = 0
        nox = float(request.form['nox'])
        rm = float(request.form['rm'])
        age = float(request.form['age'])
        dis = float(request.form['dis'])
        rad = float(request.form['rad'])
        tax = float(request.form['tax'])
        ptratio = float(request.form['ptratio'])
        b_detail = float(request.form['b_detail'])
        lstat = float(request.form['lstat'])
        x = [[crim, zn, indus, chas_detail, nox, rm,age,dis,rad,tax,ptratio,
                                        b_detail,lstat]]
        result = model.predict(data(x))

        if result < 0:
            return render_template('video.html', prediction_text="Sorry you cannot sell this house")
        else:
            return render_template('video.html', prediction_text="You Can Sell The House at {} in 1000$".format(result))

    else:
        return render_template('video.html')

def data(pred):
    housing = pd.read_csv("data.csv")
    housing_tr = my_pipeline.fit(housing.drop(columns="MEDV"))
    return my_pipeline.transform(pred)
if __name__=="__main__":
    app.run(debug=True)


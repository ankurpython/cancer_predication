import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import math
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def loadPage():
    return render_template("home.html",query="")

@app.route("/", methods=['POST'])
def cancer_predication():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/refs/heads/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    features = ["radius_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave points_mean"]
    X = df[features]
    y = df.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    # predication = model.predict(X_test)

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    features = ["radius_mean", "perimeter_mean", "area_mean", "concavity_mean", "concave points_mean"]
    new_df = pd.DataFrame(data, columns=features)
    single = model.predict(new_df)
    probability = model.predict_proba(new_df)

    if single[0] == 1:
        output1 = "The Patient is diagnosed with Cancer"
        output2 = f" - Confidence: {probability[0][1] * 100:.2f}%"
    else:
        output1 = "The Patient is not diagnosed with Cancer"
        output2 = f" - Confidence: {probability[0][0] * 100:.2f}%"

    return render_template('results.html',output1=output1,output2=output2,query1 = request.form["query1"],
                           query2 = request.form["query2"],
                           query3=request.form["query3"],
                           query4=request.form["query4"],
                           query5=request.form["query5"],
                           )

app.run()


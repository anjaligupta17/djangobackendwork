from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
# Create your views here.
def home():
    return render_template('index.html')


@app.route("/request", methods=["GET"])
def data(request):
    add = []
    Limited_Balance = int(request.GET['Limited_Balance'])
    Education = int(request.GET['Education'])
    Marriege = int(request.GET['Marriege'])
    Age = int(request.GET['Age'])
    Pay1 = int(request.GET['Pay1'])
    last_bill1 = int(request.GET['last_bill1'])
    last_bill2 = int(request.GET['last_bill2'])
    last_bill3 = int(request.GET['last_bill3'])
    last_bill4 = int(request.GET['last_bill4'])
    last_bill5 = int(request.GET['last_bill5'])
    last_bill6 = int(request.GET['last_bill6'])
    amount_paid1 = int(request.GET['amount_paid1'])
    amount_paid2 = int(request.GET['amount_paid2'])
    amount_paid3 = int(request.GET['amount_paid3'])
    amount_paid4 = int(request.GET['amount_paid4'])
    amount_paid5 = int(request.GET['amount_paid5'])
    amount_paid6 = int(request.GET['amount_paid6'])

    add = [[Limited_Balance, Education, Marriege, Age, Pay1, last_bill1, last_bill2, last_bill3, last_bill4, last_bill5,
            last_bill6, amount_paid1, amount_paid2, amount_paid3, amount_paid4, amount_paid5, amount_paid6]]

    return add

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('cleaned_data.csv')
features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
rf = RandomForestClassifier \
    (n_estimators=200, criterion='gini', max_depth=9,
     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
     max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
     min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
     random_state=4, verbose=1, warm_start=False, class_weight=None)
rf.fit(df[features_response[:-1]].values, df['default payment next month'].values)


@app.route("/request1")
def result(request1):
    l = data(request1)
    pred = rf.predict(l)[0]
    pos_prob = rf.predict_proba(l)[0][1] * 100
    neg_prob = rf.predict_proba(l)[0][0] * 100
    if pred == 1:
        x = 'The Account will be defaulted with the probability of {:.4}%. '.format(pos_prob)
    else:
        x = 'The Account will not be defaulted with the probability of {:.4}%. '.format(neg_prob)

    return render_template(request, index.html, {pred: x})
     app.run()
5
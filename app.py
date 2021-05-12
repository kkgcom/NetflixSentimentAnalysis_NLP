import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
vect = joblib.load('netflix_vector.pkl')
clf = joblib.load('netflix_svm_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # in_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(in_features)]
    inpt=request.form.get('review')
    vec= vect.transform([inpt])
    prediction = clf.predict(vec)
    output = 'good'
    if round(prediction[0],2)==1.0:
        output = 'good'
    else:
        output = 'bad'
    # output=round(prediction[0],2)
    return render_template('index.html',prediction_text='The Netflix movie review is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
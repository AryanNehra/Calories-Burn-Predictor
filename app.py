from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

model = xgb.XGBRegressor()
model.load_model('project1.model')

app=Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')

def preprocess_data(data):
    data[0] = 1 if data[0] == 'male' else 0
    data = list(map(float, data))
    return data

@app.route('/predict', methods=['POST'])
def home():
    data1=request.form['gender']
    data2=request.form['age']
    data3=request.form['height']
    data4=request.form['weight']
    data5=request.form['duration']
    data6=request.form['heartrate']
    data7=request.form['bodytemp']
    data=[data1,data2,data3,data4,data5,data6,data7]
    data=preprocess_data(data)
    arr=np.array([data])
    pred=model.predict(arr)
    return render_template('result.html',data=pred[0])


# if __name__=="__main__":
#     app.run(debug=True)
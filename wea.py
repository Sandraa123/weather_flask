from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from flask import Flask, render_template
data=pd.read_csv("daily_weather.csv")
data.dropna(inplace=True)
print(data.head(5))


x=data[['air_pressure_9am','rain_accumulation_9am']].copy()
y=data['high_humidity_3pm'].copy()

x_train, x_test, y_train, y_test=train_test_split(x,y , test_size= 0.2, random_state= 0)
lr=LinearRegression()
lr.fit(x_train,y_train)



from flask import Flask,render_template,request
wea= Flask(__name__)

@wea.route('/')
def home():
    return render_template('wea.html')


@wea.route('/pre',methods=["GET","POST"])
def hom():
    air_pressure_9am=request.form['air_pressure_9am']
    rain_accumulation_9am=request.form['rain_accumulation_9am']
    ar=np.array([air_pressure_9am,rain_accumulation_9am])
    ar=ar.astype(np.float64)
    predd=lr.predict([ar])
   


    result=predd


    return render_template('pre.html',predd=result)

if __name__ == '__main__':
    wea.run(debug=True)

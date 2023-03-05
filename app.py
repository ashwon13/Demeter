
import serial
import numpy as np

import random
import datetime as dt

arduino1 = serial.Serial("COM3",600,timeout=1)
arduino2= serial.Serial("COM4",600,timeout=1)







def getData():
    rawdata1=[]
    rawdata2=[]

    temperatureC1=0.0
    humidity1=0.0
    temperatureF1=0.0
    temperatureC2=0.0
    humidity2=0.0
    temperatureF2=0.0


    rawdata1=str(arduino1.readline())
    rawdata2=str(arduino2.readline())


    parse1 = rawdata1.split(",") #Split the data into an array called parse
    if len(parse1)==3:
        temperatureC1 = float(parse1[1])
        if parse1[0]!="":
            humidity1 = parse1[0]
            humidity1F=""
            for i in humidity1:
                if i in ["1","2","3","4","5","6","7","8","9","0","."]:
                    humidity1F+=i
            humidity1=float(humidity1F)
        

    parse2 = rawdata2.split(",") #Split the data into an array called parse
    if len(parse2)==3:
        temperatureC2 = float(parse2[1])
        if parse2[0]!="":
            humidity2 = parse2[0]
            humidity2F=""
            for i in humidity2:
                if i in ["1","2","3","4","5","6","7","8","9","0","."]:
                    humidity2F+=i
            humidity2=float(humidity2F)

    return [humidity1,temperatureC1,humidity2,temperatureC2]
        


from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('get_data')
def get_data():
    data = getData()
    humidity1=data[0]
    temperatureC1=data[1]
    humidity2=data[2]
    temperatureC2=data[3]

    emit('add_temperature1', {
        'label': 'Temperature 1',
        'backgroundColor': 'rgba(255, 0, 102, 5)',
        'borderColor': 'rgba(255, 0, 102, 1)',
        'data': [temperatureC1]
    })
    emit('add_humidity1', {
        'label': 'Humidity 1',
        'backgroundColor': 'rgba(0, 0, 255, 5)',
        'borderColor': 'rgba(0, 0, 255, 1)',
        'data': [humidity1]
    })
    emit('add_temperature2', {
        'label': 'Temperature 2',
        'backgroundColor': 'rgba(0, 153, 51, 5)',
        'borderColor': 'rgba(0, 153, 51, 5)',
        'data': [temperatureC2]
    })
    emit('add_humidity2', {
        'label': 'Humidity 2',
        'backgroundColor': 'rgba(75, 192, 192, 0.2)',
        'borderColor': 'rgba(75, 192, 192, 1)',
        'data': [humidity2]
    })
    

if __name__ == '__main__':
    socketio.run(app)



















































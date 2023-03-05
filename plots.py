from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import datetime as dt
import serial

fig2, ax2=plt.subplots()
xs=[]
ys=[]

count=0
#take a data point from serial point COM3 and plot it
def animate( xs, ys,tempC1):
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))  #Gets the exact time
    ys.append(tempC1)
    # Limit x and y lists to 50 items
    xs = xs[-50:]
    ys = ys[-50:]
    ax2.clear()
    ax2.plot(xs, ys)
    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('TMP102 Temperature over Time') #Title of Graph
    plt.ylabel('Temperature (deg C)')         #Y axis label

# Set up plot to call animate() function periodically

arduino1 = serial.Serial("COM3",9600,timeout=1)
arduino2= serial.Serial("COM4",9600,timeout=1)

while True:
    if count==100:
        plt.show()
    rawdata1=str(arduino1.readline())
    temperatureC1=0
    humidity1=0
    temperatureF1=0
    parse = rawdata1.split(",") #Split the data into an array called parse
    print(len(parse))
    if len(parse)==3:
        humidity1 = parse[0]
        humidity1F=""
        for i in humidity1:
            if i in ["1","2","3","4","5","6","7","8","9","0","."]:
                humidity1F+=i
        humidity1=float(humidity1F)
        temperatureC1 = parse[1]
        temperatureF1 = parse[2]
        temperatureF1F=""
        for i in temperatureF1:
            if i in ["1","2","3","4","5","6","7","8","9","0","."]:
                temperatureF1F+=i
        temperatureF1=float(temperatureF1F)
    count+=1
    ani = animation.FuncAnimation(fig2, animate(xs,ys,temperatureC1), fargs=(xs, ys))

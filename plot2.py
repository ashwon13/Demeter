
import serial
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import datetime as dt



fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sensor2')






xs1=[]
ys1=[]
ys2=[]
ys3=[]
ys4=[]
arduino2 = serial.Serial("COM4",600,timeout=1)

rawdata1=[]
rawdata2=[]
count=0




def animate(i, xs1, ys1,ys2,ys3,ys4):


    # Read temperature (Celsius) from TMP102
     #Must Change, input data here!!!!!!!

    rawdata1=str(arduino2.readline())
    #rawdata2=str(arduino2.readline())
    temperatureC1=0.0
    humidity1=0.0
    temperatureF1=0.0
    temperatureC2=0.0
    humidity2=0.0
    temperatureF2=0.0
    parse1 = rawdata1.split(",") #Split the data into an array called parse
    print(len(parse1),parse1)
    if len(parse1)==3:
        temperatureC1 = float(parse1[1])
        if parse1[0]!="":
            humidity1 = parse1[0]
            humidity1F=""
            for i in humidity1:
                if i in ["1","2","3","4","5","6","7","8","9","0","."]:
                    humidity1F+=i
            humidity1=float(humidity1F)
    

    # Add x and y to lists

    xs1.append(dt.datetime.now().strftime('%S.%f'))  #Gets the exact time
    ys1.append(temperatureC1)
    ys2.append(humidity1)
    # Limit x and y lists to 50 items
    xs1 = xs1[-50:]
    ys1 = ys1[-50:]
    ys2 = ys2[-50:]



    # Draw x and y lists
    ax2.clear()
    ax1.clear()
    ax2.plot(xs1, ys1,color="red")
    ax1.plot(xs1, ys2,color="blue")



    # Format plot
            #Y axis label








ani = animation.FuncAnimation(fig, animate, fargs=(xs1, ys1,ys2,ys3,ys4), interval=1)
plt.show()
    

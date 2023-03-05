from flask import Flask, render_template
import serial
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random 
from flask import Flask, render_template
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import PIL
import datetime as dt
averagerageTemp=0.0
averagerageHumidity=0.0
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Sensor1')


count=0






xs1=[]
ys1=[]
ys2=[]
ys3=[]
ys4=[]
arduino1 = serial.Serial("COM3",600,timeout=100)
arduino2=serial.Serial("COM4",600,timeout=100)

rawdata1=[]
rawdata2=[]






def animate(i, xs1, ys1,ys2,ys3,ys4,count):

    count+=1
    # Read temperature (Celsius) from TMP102
     #Must Change, input data here!!!!!!!

    rawdata1=str(arduino1.readline())
    temperatureC1=0.0
    humidity1=0.0
    temperatureF1=0.0
    temperatureC2=0.0
    humidity2=0.0
    temperatureF2=0.0
   
    
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
    count+=1
    #take a picture of the plot and save it as frame_count.png
    plt.savefig("frame_"+str(count)+".png")
    #find the average temperature and humidity
    averagerageTemp=sum(ys1)/len(ys1)
    averagerageHumidity=sum(ys2)/len(ys2)
    #print the average temperature and humidity
    print("Average Temperature: "+str(averagerageTemp))
    print("Average Humidity: "+str(averagerageHumidity))

  

    # Format plot
            #Y axis label


anim = animation.FuncAnimation(fig, animate, fargs=(xs1, ys1,ys2,ys3,ys4,count), interval=10)
print(averagerageHumidity,averagerageTemp)
plt.show()




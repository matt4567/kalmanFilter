import serial
import datetime
import time
import matplotlib.pyplot as plt
ser = serial.Serial('/dev/tty.usbmodem1421', 115200)
startTime = datetime.datetime.now();

x = []
y = []
z = []
counter = 0;
# datetime.datetime.now() - startTime <= datetime.timedelta(seconds = 1)
while (counter < 1000):
	line =  ser.readline()
	print counter
	try:
		lineSplit = line.split()
		if (len(lineSplit) != 3):
			raise Exception("Longer than 2")
		print lineSplit
		if (float(lineSplit[0]) > 10):
			raise Exception("Weird runover from prev line")
		x.append(float(lineSplit[0]))
		y.append(float(lineSplit[1]))
		z.append(float(lineSplit[2]))
	except:
		counter -= 1
	counter = counter + 1
	

ser.close()
print x[:20]
# plt.plot(x, label="X")
# plt.plot(y, label="Y")
plt.plot(z, label="Z")
plt.legend()
plt.show()
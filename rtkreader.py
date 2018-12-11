import numpy
import matplotlib.pyplot as plt
import time

file = open("rtkData.csv")

north = []
east = []
for i in file:
	values = i.split(",")
	try:
		north.append(float(values[1]))
		east.append(float(values[2]))
	except ValueError:
		if (len(north) > len(east)):
			print len(north) - len(east)
			north = north[0:len(east)]
		elif (len(north) < len(east)):
			print -len(north) + len(east)
			east = east[0:len(north)]
			
		
# 		do nothing

plt.xlabel("Relative easterly motion /m")
plt.ylabel("Relative northern motion/m")
plt.plot(east, north)
plt.show()
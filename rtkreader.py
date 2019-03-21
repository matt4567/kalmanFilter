import numpy
import matplotlib.pyplot as plt
import time

# fileName = raw_input("Which file would you like to read?  ")
fileName = "rtkDataOrientationTestOutdoors.csv"
file = open(fileName)

north = []
east = []
for n, i in enumerate(file):
	if n == 0: continue
	values = i.split(",")
	try:
		north.append(float(values[1]))
		east.append(float(values[2]))
	except ValueError: continue
		# if (len(north) > len(east)):
# 			print len(north) - len(east)
# 			north = north[0:len(east)]
# 		elif (len(north) < len(east)):
# 			print -len(north) + len(east)
# 			east = east[0:len(north)]
			
		
# 		do nothing

# beginning = 350
# end = len(north) - 1000
# print len(north)

print "northerly drift: ", max(north) - min(north), "m"

print "easterly drift:", max(east) - min(east), "m"

axes = plt.gca()
# axes.set_xlim([1.8,1.9])
# axes.set_ylim([0.55,0.65])

plt.xlabel("Relative easterly motion /m")
plt.ylabel("Relative northern motion/m")

# eastKeep = []
# northKeep =[]
# for i, val in enumerate(east):
# 	if (val < 0.014 and val > -0.0002):
# 		eastKeep.append(val)
# 		northKeep.append(north[i])


# ranges = numpy.arange(0, len(north) - 300, 100)

# for i, val in enumerate(east):
# 	if val > -3.3:
# 		east[i] -= .15

# plt.plot(east[220:320], north[220:320], 'o', label = "positions")
# print len(east)
# starts = numpy.arange(0, len(east), 100)
# finish = len(east)
# for i in starts:
# 	print i
# 	plt.plot(east[i:], north[i:], 'o', ms =1,  label = "positions")
# 	
# 
# 	# plt.plot(east[:100], north[:100], label = "ONly beginning")
# 	# plt.plot(east[500:700], north[500:700], label = "Only end")
# 	# print len(east)
# 	plt.show()
#
# rtk new antennas noise - [187:2524]
# it was on this for rtkData - east[2630:], north[2630:]
print len(east)
start = 2215
plt.plot(east[start:], north[start:], 'o', ms =1,  label = "positions")
plt.show()
 
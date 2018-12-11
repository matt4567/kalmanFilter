import matplotlib.pyplot as plt
import csv

def readData(plot):
	first = True
	counter = 0
	constantTime = 0
	times = []
	x=[]
	y=[]
	z=[]
	with open('firstrun.tsv') as tsvfile:
  		reader = csv.reader(tsvfile, delimiter='\t')
  		for row in reader:
  			counter+=1
  	# 		if counter < 2500 or counter > 4200: continue
  			if first:
  				constantTime = int(row[1])
  				print constantTime
  				first = False
  			times.append(float(row[1]) - constantTime)
  			x.append(float(row[2]))
  			y.append(float(row[3]))
  			z.append(float(row[4]))

	if plot:
		plt.plot(times, x, label="x")
		plt.plot(times, y, label="y")
		plt.plot(times, z, label="z")
		plt.legend()
		plt.xlabel("Time/ms")
		plt.ylabel("Accelleration/m/s^2")
		plt.show();

	return (times, x, y, z)

# readData(True)

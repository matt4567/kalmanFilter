import matplotlib.pyplot as plt
import time
X = []
Y = []
Z = []

file = open('FLY061.csv')
firstTime = True;
for i in file:
	if firstTime:
		firstTime = False;
		continue;
	vals = i.split(",")
# 	print vals[8]
# 	time.sleep(1)

	if (len(vals) > 10):
	
		X_val = vals[8]
		Y_val = vals[9]
		Z_val = vals[10]

	
		try:
			X.append(float(X_val))
			Y.append(float(Y_val))
			Z.append(float(Z_val))
		
		except ValueError:
			X_val += "hello"
		
print len(X)
plt.plot(X[:22000], label = "X")
plt.plot(Y[:22000], label = "Y")
plt.plot(Z[:22000], label = "Z")

plt.xlabel("Timestep")
plt.ylabel("Acceleration (g)")
plt.legend()
plt.show()


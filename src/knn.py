from sklearn import neighbors
import numpy as np
import pandas as pd

testfile="../data/test.csv"
trainfile = "../data/train.csv"

def most_common(lst):
	return max(set(lst), key=lst.count)

with open(trainfile,"r") as tr, open(testfile,"r") as te:
	a = tr.readlines()
	a.pop(0)
	b = te.readlines()
	b.pop(0)

X = []
r = []
test = []
ranges = []
with open("range.txt","r") as rangefile:
	for item in rangefile.readlines()[0].split(","):
		ranges.append(float(item))	

print(len(ranges))

for item in a:
	temp = item.strip("\n").strip().split(",")
	temp = temp[1:]
	n = []
	for i in range(len(temp)-1):
		n.append(float(temp[i])/ranges[i])
	X.append(n)
	r.append(temp[-1])

for item in b:
	temp = item.strip("\n").strip().split(",")
	temp = temp[1:]
	n = []
	for i in range(len(temp)):
		n.append(float(temp[i])/ranges[i])
	test.append(n)

"""
for line in X + test:
	for i in range(len(line)):
		if line[i] > ranges[i]:
			ranges[i] = line[i]

with open("range.txt","w") as rangetxt:
	rangetxt.write(",".join([str(x) for x in ranges]))

print(ranges)
"""


tree = neighbors.KDTree(X,leaf_size=20)

id = 0
tru = 0
for i in range(len(X)):
	dist, ind = tree.query([X[i]], k=6)
	inds = np.ndarray.tolist(ind)[0]
	if i in inds:	
		inds.remove(i)
	pred = most_common([r[y] for y in inds])
	real = r[i]
	if real == pred:
		tru += 1
	if(i%1000==0):
		print("ID= " + str(id) + ", Real value: " + real + ", Predicted value= " + pred)
	id +=1

print("Accuracy = " + str(tru*100.0/len(X) ))

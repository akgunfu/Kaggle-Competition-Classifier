from sklearn import neighbors
from random import shuffle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

def k_fold(var,val,N,i):
	if N==i:
		ret_var = var[int((len(var)/N)*(N-1)):]
		ret_val = val[int((len(var)/N)*(N-1)):]
		tra_var = var[:int((len(var)/N)*(N-1))]
		tra_val = val[:int((len(var)/N)*(N-1))]
		return ret_var, ret_val,tra_var, tra_val
	else:
		ret_var = var[int((len(var)/N)*(i-1)):int(i*len(var)/N)]
		ret_val = val[int((len(var)/N)*(i-1)):int(i*len(var)/N)]
		tra_var = var[:int((len(var)/N)*(i-1))] + var[int(i*len(var)/N):]
		tra_val = val[:int((len(var)/N)*(i-1))] + val[int(i*len(var)/N):]
		return ret_var, ret_val, tra_var, tra_val


testfile="../data/test.csv"
trainfile = "../data/train.csv"

with open(trainfile,"r") as tr, open(testfile,"r") as te:
	a = tr.readlines()
	a.pop(0)
	b = te.readlines()
	b.pop(0)

shuffle(a)

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

#Appyling dimensionality reduction
dimensions = 20
pca = PCA(n_components=dimensions)
set_new = pca.fit_transform(X+test)
x_new = set_new[:len(X)]
test_new = set_new[len(X):]

#Init neural network
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,15), random_state=1)

#Tenfold training and validation

totes = 0.0

t_set = np.ndarray.tolist(x_new)

for i in range(1,11):
	te_vars,te_vals,tr_vars,tr_vals = k_fold(t_set,r,10,i)
	print("Training Neural Network for fold number " + str(i))
	clf.fit(tr_vars,tr_vals)
	print("Predicting for fold number " + str(i))
	preds = clf.predict(te_vars)
	count = 0
	for b in range(len(preds)):
		if te_vals[b] == preds[b]:
			count+=1
	acc = count*100.0/len(preds)
	print("Predicted fold " + str(i) + " with %"+ str(acc) + " accuracy for " + str(dimensions) + " dimensions.")
	totes += acc

print("Averaged accuracy of " + str(totes/10) + " for training data.")


#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

data_dict.pop('TOTAL', 0) #this remove the "TOTAL" outlier

features = ["salary", "bonus"]


data = featureFormat(data_dict, features)

#data = np.delete(data, (67), axis=0) #This line deletes the row 67, which was the outlier


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

features, target = targetFeatureSplit( data )

#To identy the name of the outlier

mvalue = max(target)

print(mvalue) #Print the max value

max_value = target.index(max(target))

print(max_value) #Print the index of the max value (It used to be 67, to see it dete line 20)

#This holy line prints me the names of the guys that owns a salary over a 1M and bonuses over 5M
for k,v in data_dict.items():
	salary=float(v.get("salary"))
	bonus=float(v.get("bonus"))
	if(salary>1000000 and bonus>=5000000):
		print(k)

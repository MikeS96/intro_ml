#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#clf = SVC(kernel="linear", C=1.0, gamma=0.1)
#clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000) #with C = 10 is 0.616040955631 - #with C = 100 is 0.616040955631
#with C = 1000 is 0.821387940842 #with C = 10000 is 0.892491467577

#features_train = features_train[:len(features_train)/100] #This lines reduce the train set to 1%
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train) 
print "training time:", round(time()-t0, 3), "s" 

t1 = time()
result = clf.predict(features_test) #Here we hive a new point and the classifier has to predict
print "Predict time:", round(time()-t1, 3), "s"

print('The accuracy is')
print(accuracy_score(labels_test, result))

u = 0
v = 0

for x in range(len(result)): #For to know the number of mails for chris
	if result[x] == 1:
		u = u + 1
	else:
		u = u
		
for y in result: #For to know the number of mails for chris
	if y == 1:
		v = v + 1
	else:
		v = v
		
print(u)
print(v)
print(len(result))
print(result[26])
print(result[50])

#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from time import time

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()

t0 = time()
clf = clf.fit(X_train, y_train)
print "training time:", round(time()-t0, 3), "s" 

t1 = time()
result = clf.predict(X_test)
print "Predict time:", round(time()-t1, 3), "s"

print('The accuracy is')
print(accuracy_score(y_test, result))

#the number of Pois in the test set is: 
cont = 0
for x in y_test:
	if x == 1:
		cont = cont + 1
print('The number of Pois in the test set is: ')
print cont

from sklearn.metrics import precision_score

print confusion_matrix(y_test, result)

print 'The precision of the method id: '
prec = precision_score(y_test, result, average=None)  
print prec

print 'The recall of the method id: '
rec = recall_score(y_test, result, average=None)  
print rec

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

prec = precision_score(true_labels, predictions, average='macro')  
print prec

rec = recall_score(true_labels, predictions, average='macro')  
print rec
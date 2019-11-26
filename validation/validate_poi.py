#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from time import time

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


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


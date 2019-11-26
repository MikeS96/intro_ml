#!/usr/bin/python

#I based my work on this guy, who made a great work. https://htmlpreview.github.io/?https://github.com/capn-freako/IPython_Notebooks/blob/master/Udacity_Machine_Learning_Intro-Final_Project.html#task1

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from matplotlib     import pyplot as plt
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Here i select the features that im going to use
features_list = ['poi',
        'salary',
        'deferral_payments',
        'total_payments',
        'bonus',
        'total_stock_value',
        'loan_advances',
        ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##########################################################################
### Task 2: Remove outliers
# The data contains a TOTAL sample, which will confuse our classifier if we don't eliminate it.
data_dict.pop('TOTAL', 0)

import pandas as pd
from matplotlib.colors import ListedColormap

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


# Note: It appears that pandas.scatter_matrix doesn't quite work
#       as advertised, in the documentation. If it did, this wouldn't
#       be necessary. You could pass a colormap, instead.
palette = {0 : 'blue', 1 : 'red'}
###the next function maps all the data in labels to the variable labels_c
###It gives a color to the labels based on the pallete, if it is zero the color is blue
### if it is one the color is red. thus all this data is collected in labels_c and used 
###To do the graph
labels_c = map(lambda x: palette[int(x)], labels)

###Create a dataframe with pandas with the features
data_frame = pd.DataFrame(features, columns=features_list[1:])
grr = pd.plotting.scatter_matrix(data_frame, alpha=0.8, c=labels_c)
plt.show()



from operator   import itemgetter

###This line is heavy af and shows the top 3 values of each feature
for feature in features_list[1:]:
    l = [(item[0], item[1][feature]) for item in data_dict.items() if not np.isnan(float(item[1][feature]))]
    l.sort(key=itemgetter(1), reverse=True)
    print "Top 3 values for feature, '{}': {}".format(feature, l[:3])
    
###These features appear to be pretty useeless, so they are goung to be removed
for feature in ['loan_advances', 'total_payments']:
    features_list.remove(feature)

##########################################################################

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

###This new features are going to be multiply with the previous one to create consistents ones
###The new feature consist in the multiplication of all the 4 new features

constituent_features_list = ['poi',
        'shared_receipt_with_poi',
        'expenses',
        'from_this_person_to_poi',
        'from_poi_to_this_person',
        ]

###Here he did the same than before, create the data dict and data and plot it to see the correlation
new_data = featureFormat(data_dict, constituent_features_list)
new_labels, new_features = targetFeatureSplit(new_data)
new_labels_c = map(lambda x: palette[int(x)], new_labels)

new_data_frame = pd.DataFrame(new_features, columns=constituent_features_list[1:])
grr = pd.plotting.scatter_matrix(new_data_frame, alpha=0.8, c=new_labels_c)

plt.show()


for key in data_dict.keys():
    features_dict = data_dict[key]
    res = 1
    for subkey in constituent_features_list[1:]:
        x = features_dict[subkey]
        if(np.isnan(float(x))):
            res = 0
        else:
            res *= x
    data_dict[key]['expenses_and_poi_contact'] = res

features_list.append('expenses_and_poi_contact')

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

labels_c = map(lambda x: palette[int(x)], labels)

data_frame = pd.DataFrame(features, columns=features_list[1:])
grr = pd.scatter_matrix(data_frame, alpha=0.8, c=labels_c)

plt.show()

###With this i can itendify the outlier easily

for feature in ['expenses_and_poi_contact']:
    l = [(item[0], item[1][feature]) for item in data_dict.items() if not np.isnan(float(item[1][feature]))]
    l.sort(key=itemgetter(1), reverse=True)
    print "Top 3 values for feature, '{}': {}".format(feature, l[:3])


##########################################################################


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


from sklearn                  import svm, tree
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.metrics          import precision_score, recall_score

sys.path.append("ud120-projects/final_project/")
#This line is really cool because it gives all the result of all the classifiers at once
from tester                   import test_classifier

print "Trying the SVM classifier..."
clf = svm.SVC()
test_classifier(clf, data_dict, features_list)

print "\nTrying the Decision Tree classifier..."
clf = tree.DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list)

print "Trying the AdaBoost classifier..."
clf = AdaBoostClassifier()
test_classifier(clf, data_dict, features_list)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

###All this is to tune the classifier
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics         import classification_report

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, stratify=labels)

# Set the parameters by cross-validation
tuned_parameters = [
    {'criterion' : ['gini', 'entropy'],
     'splitter'  : ['best', 'random'],
    },
]

scores = ['precision', 'recall']

for score in scores:
    print("Tuning hyper-parameters for %s:" % score)

    clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters,
                       scoring='%s_macro' % score)
    clf.fit(features_train, labels_train)

    print("\tBest parameters set found on development set:"),
    print(clf.best_params_)
    print("\tGrid scores on development set:")

    '''
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("\t\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	'''
	
    print("\tDetailed classification report:")
    y_true, y_pred = labels_test, clf.predict(features_test)
    ###This line preints all the metrics involved with the work, is really nice
    print(classification_report(y_true, y_pred))

'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''
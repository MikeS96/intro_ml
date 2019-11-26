#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


print len(features_train)
### your code goes here

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features_train, labels_train)


print('The accuracy for the test set is: ')
print clf.score(features_test,labels_test)

#To get the most important features and its lenght

features_importants = clf.feature_importances_

print 'The number of important features is: '
print len(features_importants)

number_of_ourliers = 0
cont = 0
val = 0
highest_value = 0
for x in features_importants:
	if x > 0.2:
		if highest_value < x:
			highest_value = x
			val = cont
			number_of_ourliers = number_of_ourliers + 1
	cont = cont + 1

print 'The highest Threshfold for the important feature is: '
print highest_value

print 'The index of the highest value is: '
print val

print 'the name of the most important feature is: '
names = vectorizer.get_feature_names()
print names[val]

print 'the name of outliers is (values above > 0.2): '
print number_of_ourliers
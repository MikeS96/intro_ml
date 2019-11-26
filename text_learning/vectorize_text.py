#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset

        #This lines were commented and iddented
        #temp_counter += 1
        #if temp_counter < 200:
        path = os.path.join('..', path[:-1])
        print path
        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email
        steemed_mail = parseOutText(email) #I pass each email to the paseOutText that steem the mails
        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]
        steemed_mail = steemed_mail.replace("sara", "")
        steemed_mail = steemed_mail.replace("shackleton", "")
        steemed_mail = steemed_mail.replace("chris", "")
        steemed_mail = steemed_mail.replace("germani", "")  

        steemed_mail = steemed_mail.replace("sshacklensf", "") #This part was added due to lesson 12, it is an outlier\
        steemed_mail = steemed_mail.replace("cgermannsf", "") #This part was added due to lesson 12, it is an outlier

        ### append the text to word_data
        word_data.append(steemed_mail)
        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
        if name == "sara": 
            from_data.append(0)
            #this can also be made by
            #from_data.append(0)

        if name  == "chris":
            from_data.append(1)

        
        email.close()

print "emails processed"

#Prints the data contained in the 152 position
print 'The phrase contained in the position 152 is: '
print word_data[152]

print len(from_data)
print len(word_data)

from_sara.close()
from_chris.close()


pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )


### in Part 4, do TfIdf vectorization here

#I apply a Tfidt here
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(word_data)
print 'the number of features extrated by Tfidf is: '
print len(vectorizer.get_feature_names()) #this line prints the number of features extracted by Tfidf


features = vectorizer.get_feature_names()

print 'The features extracted in the position 34597 is: '
print features[34597]


#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print(len(enron_data)) #len(A) the number of columns Number of persons in the dataset (146 persons)

print(len(enron_data["DONAHUE JR JEFFREY M"])) #Number of features for a specific person (21 features)

print((enron_data["DONAHUE JR JEFFREY M"]["poi"])) #The poi shows the persons of interest 
persons_int = 0

for x in enron_data: #For loop to look all the persons of interest in the dataset
	if(enron_data[x]["poi"]==True):
		persons_int = persons_int + 1

print("The number persons of interest are")
print(persons_int)

poi_name_record = open("../final_project/poi_names.txt").read().split("\n")  #This line say the number of Persons of interest in the txt which basiclly are all of them
poi_name_total = [record for record in poi_name_record if "(y)" in record or "(n)" in record]
print("Total number of POIs: ", len(poi_name_total))

print((enron_data["DONAHUE JR JEFFREY M"]))

print((enron_data["PRENTICE JAMES"]["total_stock_value"])) #The poi shows the persons of interest   total_stock_value   

print((enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]))

print((enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]))   

print((enron_data["LAY KENNETH L"]["total_payments"]))   

print((enron_data["SKILLING JEFFREY K"]["total_payments"])) 

print((enron_data["FASTOW ANDREW S"]["total_payments"])) 

non_salary = 0
email = 0
pay = 0
for x in enron_data: #For loop to look all the persons of interest in the dataset
	if(enron_data[x]["salary"]=="NaN"):
		non_salary = non_salary + 1
	if(enron_data[x]["email_address"]=="NaN"):
		email = email + 1 
	if(enron_data[x]["total_payments"]=="NaN"):
		pay = pay + 1 

print("The number persons with quantifies salary")
print(146-non_salary)
print("The number persons with quantifies email")
print(146-email)
print("The number persons with quantifies email")
print(pay)

persons_int = 0
for x in enron_data: #For loop to look all the persons of interest in the dataset
	if(enron_data[x]["poi"]==True and enron_data[x]["total_payments"]=="NaN"):
		persons_int = persons_int + 1

print("The number persons of interest are")
print(persons_int)

#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []

    residual_error = abs(predictions -net_worths) ** 2
    #print sorted(residual_error) #This methodsort the list from minimun to max value

    cleaned_data = list(zip(ages, net_worths, residual_error))
    # print [row[2] for row in cleaned_data] #This for gives the row of a list

    for i in range(9):
        residual_list = [row[2] for row in cleaned_data] #Extract the secound column of the list
        max(residual_list)  # obtain its maximun value
        ind = residual_list.index(max(residual_list)) #Extract the index of the max value of the secound column
        cleaned_data.pop(ind) #this deletes the element in position num of the index

    print(len(cleaned_data))
    

    return cleaned_data


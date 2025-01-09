#!/usr/bin/env python
# coding: utf-8

# In[4]:


# User defined functions
def mean_value(*n):
    # Initialize the sum of the numbers to 0
    sum = 0
    # Initialize a counter to keep track of the number of elements
    counter = 0
    # Iterate through each number in the input
    for x in n:
        # Increment the counter for each number
        counter = counter + 1
        # Add the current number to the sum
        sum += x
    # Calculate the mean by dividing the total sum by the count of numbers
    mean = sum / counter
    # Return the calculated mean value
    return mean


# In[6]:


# Find the product of given numbers

# Defining the function that takes a variable number of arguments
def product(*n):
    # Initialize the result variable to 1 (multiplicative identity)
    result = 1
    
    # Iterate over the length of the input arguments
    for i in range(len(n)):
        # Multiply each argument with the result
        result *= n[i]
    
    # Return the final product
    return result


# In[10]:


# Calling the function "product" with n number of arguments
product(20, 30, 40, 50)


# In[ ]:





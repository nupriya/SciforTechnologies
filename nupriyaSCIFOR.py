#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# Question: Reverse a string using a for loop in Python.

# In[2]:


def reverse_string(str):  
    str1 = ""   # Declaring empty string to store the reversed string  
    for i in str:  
        str1 = i + str1  
    return str1    # It will return the reverse string to the caller function  
     
str = "NUPRIYASAXENA"    # Given String       
print("The original string is: ",str)  
print("The reverse string is",reverse_string(str)) # Function call  


# Question: Write a Python program to find the sum of all numbers in a list using a for loop.

# In[3]:


lst = []
num = int(input('How many numbers: '))
for n in range(num):
    numbers = int(input('Enter number '))
    lst.append(numbers)
print("Sum of elements in given list is :", sum(lst))


# Question: Write a Python program that checks whether a given number is even or odd using an if-else statement.

# In[7]:


num = int (input('Enter any number to test whether it is odd or even:'))

if (num % 2) == 0:

              print ('The number is even')

else:

              print ('The provided number is odd')


# Question: Implement a program to determine if a year is a leap year or not using if-elif-else statements.

# In[9]:


def CheckLeap(Year):  
   
  if((Year % 400 == 0) or  
     (Year % 100 != 0) and  
     (Year % 4 == 0)):   
    print("Given Year is a leap Year")  
   
  else:  
    print ("Given Year is not a leap Year")
Year = int(input("Enter the number: "))    
CheckLeap(Year)  


# Question: Use a lambda function to square each element in a list
# 

# In[11]:


nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Original list of integers:")
print(nums)
print("\nSquare every number of list:")

square_nums = list(map(lambda x: x ** 2, nums))
print(square_nums)


# Question: Write a lambda function to calculate the product of two

# In[12]:



product = lambda a, b : a * b


a = 5
b = 7

product = a * b

print("The product of number:", product)


# In[ ]:





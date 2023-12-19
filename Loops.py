#!/usr/bin/env python
# coding: utf-8

# # Looping

# A loop is a programming structure that allows you to repeat a set of instructions until a specific condition is met.
# 
# This is useful for tasks that need to be repeated multiple times
# 
# Loop change the flow control of program that is why it is also called control structures
# 

# #### Types of loops

# There are mainly three types of loops. Let’s discuss them one by one.
# 
# 1-For loop
# 
# 2-While loop
# 
# 3-Nested loop
# 

# ### For loop

# A for loop in Python is used to iterate over a sequence.
# a control flow statement that is used to repeatedly execute a group of statements as long as the condition is satisfied.
# 

# In[3]:


numbers = [1, 2, 3, 4, 5]
for i in numbers:
    print(i)


# #### While loop

# used to repeat a specific block of code an unknown number of times, until a condition is met.
#  while loop is a control flow statement that allows code to be executed repeatedly based on a given Boolean condition.
# 

# In[5]:


count = 0
while (count < 3):
    count = count + 1
    print("Hello")


# ### Nested loop

# A nested loop means a loop statement inside another loop statement. That is why nested loops are also called “loop inside loops“.
#         The outer loop runs first, and each time it runs, the inner loop runs as well.

# In[11]:


x = [1, 2]
y = [4, 5]
 
for i in x:
    for j in y:
        print(i, j)


# In[ ]:





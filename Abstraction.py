#!/usr/bin/env python
# coding: utf-8

# # Abstraction

# It hides unnecessary code details from the users.
# 
# data abstraction in python can be achived by creating abstract classes.

# In[1]:


from abc import ABC,abstractmethod


# In[2]:


class employee(ABC):
    def emp_id(self,id,name,age):
        pass
class childemployee(employee):
    def emp_id(self,id):
        print("emp_id is 1")
        
emp1= childemployee()
emp1.emp_id(id)


# In[ ]:





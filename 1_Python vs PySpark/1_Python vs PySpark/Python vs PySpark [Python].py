#!/usr/bin/env python
# coding: utf-8

# # Python vs PySpark Commands
# *Python*
#
#
# ### Create a Pandas dataframe
#
# In Python, if we want to create a dataframe, we need to import a library. Pandas is the most common library used for this. Let's create one now.

# In[1]:


import pandas as pd

# initialize list of lists
data = [["tom", 10], ["nick", 15], ["juli", 14]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=["Name", "Age"])


# ## Display the dataframe and it's properties

# In[3]:


df.head(5)


# In[4]:


# Get a list of columns
df.columns


# In[5]:


# How many rows are in the dataframe?
len(df)


# ## Read in data

# In[10]:


path = "students.csv"
df = pd.read_csv(path)
df


# ## Aggregate Data

# In[12]:


df.groupby("gender").agg({"math score": ["mean", "min", "max"]})


# In[ ]:

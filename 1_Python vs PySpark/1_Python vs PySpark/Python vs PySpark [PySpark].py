#!/usr/bin/env python
# coding: utf-8

# # Python vs PySpark Commands
# *PySpark*
#
#
# ## Creating a Spark Session

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("PySpark").getOrCreate()
spark


# ### Create a Spark dataframe
#
# In PySpark you need to create a Spark session first before you do anything. Then the createdataframe is inherent in session.

# In[2]:


# initialize list of lists (same as in python)
data = [["tom", 10], ["nick", 15], ["juli", 14]]

# Create the pandas DataFrame
df = spark.createDataFrame(data, ["Name", "Age"])


# ## Display Dataframe and it's properties

# In[3]:


df.show()


# In[4]:


# This is closer to pandas df.head()
df.toPandas()


# In[11]:


# View column names
# This is the same
df.columns


# In[5]:


# How many rows are in the dataframe
df.count()


# ## Read in data

# In[7]:


path = "students.csv"
df = spark.read.csv(path, header=True)
df.toPandas()


# ## Aggregate Data
#
# This is method is very similar to pandas but you can only do one metric at a time

# In[8]:


df.groupBy("gender").agg({"math score": "mean"}).show()


# For more than one aggreate... use this

# In[42]:


from pyspark.sql import functions as F

df.groupBy("gender").agg(
    F.min("math score"), F.max("math score"), F.avg("math score")
).show()


# ## Sparks Immutability
#
# Spark DataFrame's are built on top of RDDs which are immutable in nature, hence Data frames are immutable in nature as well.
#
# So if you make a change to a dataframe like adding a column or changing any of the values in the dataframe using the same naming convention, it will generate a new dataframe (with a new unique ID) instead of updating the existing data frame.

# In[9]:


# Let's fetch the id of our dataframe we created above
df.rdd.id()


# In[10]:


# Even if we duplicate the dataframe, the ID remains the same
df2 = df
df2.rdd.id()


# In[11]:


# It's not until we change the df in some way, that the ID changes
df = df.withColumn("new_col", df["math score"] * 2)
df.rdd.id()


# ## Spark's Lazy Comuptation
#
# What does that mean exactly?
#
# As the name itself indicates its definition, lazy evaluation in Spark means that the execution will not start until it absolutuley HAS to.
#
# Let's look at an example.

# In[12]:


# These kinds of commands won't actually be run...
df = df.withColumn("new_col", df["math score"] * 2)


# In[13]:


# Until we executute a command like this
collect = df.collect()


# In[14]:


# Or this
print(df)


# So you can think of Spark like a lazy teenager who doesn't have to clean his room until you come an inspect it :)
#
# The benefit is saving resources and optimizing the Spark cluster overall.

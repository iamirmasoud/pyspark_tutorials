#!/usr/bin/env python
# coding: utf-8

# # Search and Filter DataFrames in PySpark
#
# Once we have created our Spark Session, read in the data we want to work with and done some basic validation, the next thing you'll want to do is start exploring your dataframe. There are several option in PySpark to do this, so we are going to start with the following in this lecture, and continue to dive deeper in the next several lectures.
#
# ### Agenda:
#
#  - Introduce PySparks SQL funtions library
#  - Select method
#  - Order By
#  - Like Operator (for searching a string)
#  - Substring Search
#  - Is In Operator
#  - Starts with, Ends with
#  - Slicing
#  - Filtering
#  - Collecting Results as Objects
#
# Let's get started!

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Select").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in the DataFrame for this Notebook

# In[3]:


path = "Datasets/"
fifa = spark.read.csv(path + "fifa19.csv", inferSchema=True, header=True)


# ## About this dataframe
#
# The **fifa19.csv** dataset includes a list of all the FIFA 2019 players and their attributes listed below:
#
#  - **General**: Age, Nationality, Overall, Potential, Club
#  - **Metrics:** Value, Wage
#  - **Player Descriptive:** Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight
#  - **Possition:** LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB,
#  - **Other:** Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.
#
# **Source:** https://www.kaggle.com/karangadiya/fifa19

# In[4]:


# Take a look at the first few lines
fifa.limit(4).toPandas()


# In[5]:


print(fifa.printSchema())


# ## Select
# There are a variety of functions you can import from pyspark.sql.functions. Check out the documentation for the full list available:
# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions

# In[3]:


# Import the functions we will need:
from pyspark.sql.functions import *

# countDistinct,avg,stddev
# abs # Absolute value
# acos # inverse cosine of col, as if computed by java.lang.Math.acos()


# Since this is a sql function, the calls are pretty intuitive....

# In[6]:


fifa.select(["Nationality", "Name", "Age"]).show(5)


# **Order By**

# In[7]:


# who is the youngest player in the dataset?
fifa.select(["Nationality", "Name", "Age"]).orderBy("Age").show(5)


# In[8]:


# Who is the oldest player?
fifa.select(["Nationality", "Name", "Age"]).orderBy(fifa["Age"].desc()).show(5)


# **Like**

# In[9]:


# If we wanted to look for all players that had "Barcelona" in their club title
# We could use the like operator
fifa.select("Name", "Club").where(fifa.Club.like("%Barcelona%")).show(5, False)


# **Substrings**
#
# .substr(starting postion,length)
#
# Use this if you want to return a particular portion within a string

# In[10]:


# Select last 4 characters of the photo column to understand all file types used
# This says return
fifa.select("Photo", fifa.Photo.substr(-4, 4)).show(5, False)


# In[11]:


# Or we could get the date that the string of numbers there
fifa.select("Photo", fifa.Photo.substr(32, 11)).show(5, False)


# **ISIN**
#
# You can also use ISIN to search for a list of options within a column.

# In[5]:


fifa[fifa.Club.isin("FC Barcelona", "Juventus")].limit(4).toPandas()


# **Starts with Ends with**
#
# Search for a specific case - begins with "x" and ends with "x"

# In[12]:


fifa.select("Name", "Club").where(fifa.Name.startswith("L")).where(
    fifa.Name.endswith("i")
).limit(4).toPandas()


# #### Slicing a Dataframe

# In[ ]:


# Starting
print("Starting row count:", fifa.count())
print("Starting column count:", len(fifa.columns))

# Slice rows
df2 = fifa.limit(300)
print("Sliced row count:", df2.count())

# Slice columns
cols_list = fifa.columns[0:5]
df3 = fifa.select(cols_list)
print("Sliced column count:", len(df3.columns))


# **Slicing Method**
#
# pyspark.sql.functions.slice(x, start, length)[source] <br>
# Returns an array containing all the elements in x from index start (or starting from the end if start is negative) with the specified length.  <br>
# <br>
# *Note: indexing starts at 1 here*

# In[13]:


# This is within an array
from pyspark.sql.functions import slice

df = spark.createDataFrame([([1, 2, 3],), ([4, 5],)], ["x"])
df.show()
df.select(slice(df.x, 2, 2).alias("sliced")).show()


# If we want to simply slice our dataframe (ie. limit the number of rows or columns) we can do this...

# ## Filtering Data
#
# A large part of working with DataFrames is the ability to quickly filter out data based on conditions. Spark DataFrames are built on top of the Spark SQL platform, which means that is you already know SQL, you can quickly and easily grab that data using SQL commands, or using the DataFram methods (which is what we focus on in this course).

# In[18]:


fifa.filter("Overall>50").limit(4).toPandas()


# In[19]:


# Using SQL with .select()
fifa.filter("Overall>50").select(["ID", "Name", "Nationality", "Overall"]).limit(
    4
).toPandas()


# **Try it yourself!**
#
# Edit the line below to select only closing values above 800

# In[22]:


# Try it yourself!
# Edit the line below to select only overall scores of LESS THAN 80
fifa.filter("Overall<80").select(["ID", "Name", "Nationality", "Overall"]).limit(
    4
).toPandas()


# In[65]:


fifa.select(["Nationality", "Name", "Age", "Overall"]).filter("Overall>70").orderBy(
    fifa["Overall"].desc()
).show()


# ### Collecting Results as Objects
#
# The last thing we need to cover is collecting results as objects. If we wanted to say print individual names from an output, we need to essentially remove the item from the dataframe into an object. Like this

# In[14]:


# Collecting results as Python objects
# you need the ".collect()" call at the end to "collect" the results
result = (
    fifa.select(["Nationality", "Name", "Age", "Overall"])
    .filter("Overall>70")
    .orderBy(fifa["Overall"].desc())
    .collect()
)


# In[12]:


# Note the nested structure returns a nested row object
type(result[0])


# If we want to call on these results it would look something like this...
#
# *Think of it like a matrix, first number is the row number and the second is the column number*

# In[13]:


print("Best Player Over 70: ", result[0][1])
print("Nationality of Best Player Over 70: ", result[0][0])
print("")
print("Worst Player Over 70: ", result[-1][1])
print("Nationality of Worst Player Over 70: ", result[-1][0])


# Rows can also be called to turn into dictionaries if needed

# In[51]:


row.asDict()


# Or iterated over like this...

# In[52]:


for item in result[0]:
    print(item)


# Check out this link for more info on other methods:
# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark-sql-module

# ### Great job! That's it!

#!/usr/bin/env python
# coding: utf-8

# # Handling Missing Data in PySpark HW Solutions
#
# In this HW assignment you will be strengthening your skill sets dealing with missing data.
#
# **Review:** you have 2 basic options for filling in missing data (you will personally have to make the decision for what is the right approach:
#
# 1. Drop them missing data points (including the entire row)
# 2. Fill them in with some other value.
#
# Let's practice some examples of each of these methods!
#
#
# #### But first!
#
# Start your Spark session

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("nulls").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in the dataset for this Notebook

# In[68]:


df = spark.read.csv("Datasets/Weather.csv", inferSchema=True, header=True)


# ## About this dataset
#
# **New York City Taxi Trip - Hourly Weather Data**
#
# Here is some detailed weather data for the New York City Taxi Trips.
#
# **Source:** https://www.kaggle.com/meinertsen/new-york-city-taxi-trip-hourly-weather-data

# ### Print a view of the first several lines of the dataframe to see what our data looks like

# In[69]:


df.limit(8).toPandas()


# ### Print the schema
#
# So that we can see if we need to make any corrections to the data types.

# In[70]:


print(df.printSchema())


# ## 1. How much missing data are we working with?
#
# Get a count and percentage of each variable in the dataset to answer this question.

# In[71]:


from pyspark.sql.functions import *


def null_value_calc(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if nullRows > 0:
            temp = k, nullRows, (nullRows / numRows) * 100
            null_columns_counts.append(temp)
    return null_columns_counts


null_columns_calc_list = null_value_calc(df)
spark.createDataFrame(
    null_columns_calc_list,
    ["Column_With_Null_Value", "Null_Values_Count", "Null_Value_Percent"],
).show()


# ## 2. How many rows contain at least one null value?
#
# We want to know, if we use the df.ha option, how many rows will we loose.

# In[74]:


og_len = df.count()
drop_len = df.na.drop().count()
print("Total Rows in the DF: ", og_len)
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# Yikes! Everything

# ## 3. Drop the missing data
#
# Drop any row that contains missing data across the whole dataset

# In[75]:


dropped = df.na.drop()
dropped.limit(4).toPandas()

# Note this statement is equivilant to the above:
# zomato.na.drop(how='any').limit(4).toPandas()


# Yep, we have no more data :(

# ## 4. Drop with a threshold
#
# Count how many rows would be dropped if we only dropped rows that had a least 12 NON-Null values

# In[83]:


og_len = df.count()
drop_len = df.na.drop(thresh=12).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# ## 5. Drop rows according to specific column value
#
# Now count how many rows would be dropped if you only drop rows whose values in the tempm column are null/NaN

# In[84]:


og_len = df.count()
drop_len = df.na.drop(subset=["tempm"]).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[88]:


# Another way to do the above
og_len = df.count()
drop_len = df.filter(df.tempm.isNotNull()).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# ## 6. Drop rows that are null accross all columns
#
# Count how many rows would be dropped if you only dropped rows where ALL the values are null

# In[85]:


og_len = df.count()
drop_len = df.na.drop(how="all").count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# That's good news!

# ## 7. Fill in all the string columns missing values with the word "N/A"
#
# Make sure you don't edit the df dataframe itself. Create a copy of the df then edit that one.

# In[89]:


null_fill = df.na.fill("N/A")
null_fill.limit(4).toPandas()


# ## 8. Fill in NaN values with averages for the tempm and tempi columns
#
# *Note: you will first need to compute the averages for each column and then fill in with the corresponding value.*

# In[91]:


def fill_with_mean(df, include=set()):
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    #     stats = stats.select(*(col(c).cast("int").alias(c) for c in stats.columns)) #IntegerType()
    return df.na.fill(stats.first().asDict())


updated_df = fill_with_mean(df, ["tempm", "tempi"])
updated_df.limit(5).toPandas()


# ### That's it! Great Job!

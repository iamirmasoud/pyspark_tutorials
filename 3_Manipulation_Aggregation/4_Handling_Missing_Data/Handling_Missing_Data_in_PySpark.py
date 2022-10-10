#!/usr/bin/env python
# coding: utf-8

# # Handling Missing Data in PySpark
#
# In the real world, most datasets you work with will be incomplete, which means you will have missing data. You have 2 basic options for filling in missing data (you will personally have to make the decision for what is the right approach):
#
# 1. Drop the missing data points (including the possbily the entire row)
# 2. Fill them in with some other value (like the average).
#
# There are also two different types of missing data to be aware of:
#
# 1. null values represents "no value" or "nothing", it's not even an empty string or zero. It can be used to represent that nothing useful exists.
# 2. NaN stands for "Not a Number", it's usually the result of a mathematical operation that doesn't make sense, e.g. 0.0/0.0
#
# Let's cover examples of each of these methods!

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("nulls").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in data for this Notebook

# In[2]:


# Start by reading a basic csv dataset
# Let Spark know about the header and infer the Schema types!

# Some csv data
zomato = spark.read.csv("Datasets/zomato.csv", inferSchema=True, header=True)


# ## About this dataset
#
# This dataset contains the aggregate rating of restaurant in Bengaluru India from Zomato.
#
# **Source:** https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants

# In[3]:


print(zomato.printSchema())


# In[4]:


from pyspark.sql.functions import *

# Edit some var types
from pyspark.sql.types import *

df = zomato.withColumn(
    "approx_cost(for two people)",
    zomato["approx_cost(for two people)"].cast(IntegerType()),
).withColumn("votes", zomato["votes"].cast(IntegerType()))
# QA
print(df.printSchema())


# In[5]:


df.limit(4).toPandas()


# Note that nulls values appear as "None" in the Pandas print out above. If we show the null values for the cuisines variable in attempt to view that first restaurant "Jalsa", we can see it appear as "null" below.

# In[6]:


from pyspark.sql import functions as F

# zomato.filter("cuisines='None'").agg(F.count(zomato.name)).show()
df.filter(df.cuisines.isNull()).select(["name", "cuisines"]).show(5)


# ## Missing Data Statistics
#
# It is always valualuable to know how much missing data you are going to be working with before you take any action like filling missing values with an average or dropping rows completly. Here is a good script to get you started. We will also explore more later on in this notebook.

# In[1]:


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
    null_columns_calc_list, ["Column_Name", "Null_Values_Count", "Null_Value_Percent"]
).show()


# In[ ]:


# Another way if you prefer
from pyspark.sql.functions import col, count, isnan, when

# first row: null count
nulls = df.select(
    [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
)
# Second row: null percent
percent = df.select(
    [
        format_number(
            ((count(when(isnan(c) | col(c).isNull(), c)) / df.count()) * 100), 1
        ).alias(c)
        for c in df.columns
    ]
)

result = nulls.union(percent)

result.toPandas()


# ## Drop the missing data
#
# PySpark has a really handy .na function for working with missing data. The drop command has the following parameters:
#
#     df.na.drop(how='any', thresh=None, subset=None)

# In[ ]:


# Drop any row that contains missing data across the whole dataset
df.na.drop().limit(4).toPandas()

# Note this statement is equivilant to the above:
# df.na.drop(how='any').limit(4).toPandas()


# In[ ]:


# Of course you will want to know how many rows that affected before you actually execute it..
og_len = df.count()
drop_len = df.na.drop().count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# Woah! 88% is a lot! We better figure out a better method.

# In[ ]:


# Drop rows that have at least 8 NON-null values
og_len = df.count()
drop_len = df.na.drop(thresh=8).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# Way better!

# In[ ]:


# Only drop the rows whose values in the sales column are null
og_len = df.count()
drop_len = df.na.drop(subset=["votes"]).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[ ]:


# Another way to do the above
og_len = df.count()
drop_len = df.filter(df.rate.isNotNull()).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[ ]:


# Drop a row only if ALL its values are null.
og_len = df.count()
drop_len = df.na.drop(how="all").count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# ## Fill the missing values
#
# We can also fill the missing values with new values. If you have multiple nulls across multiple data types, Spark is actually smart enough to match up the data types. For example:

# In[ ]:


# Fill all nulls values with one common value (character value)
df.na.fill("MISSING").limit(4).toPandas()


# In[ ]:


# Fill all nulls values with one common value (numeric value)
df.na.fill(999).limit(10).toPandas()


# Usually you should specify what columns you want to fill with the subset parameter

# In[ ]:


df.filter(df.name.isNull()).na.fill("No Name", subset=["name"]).limit(5).toPandas()


# A very common practice is to fill values with the **mean value** for the column. Here is a fun function to that in an automatted way.

# In[ ]:


def fill_with_mean(df, include=set()):
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    return df.na.fill(stats.first().asDict())


updated_df = fill_with_mean(df, ["votes"])
updated_df.limit(5).toPandas()


# ## Keeping the missing data
# A few machine learning algorithms can easily deal with missing data. Just do your research and make sure the nulls values are not impacting the integrity of your analysis.

# That is all we need to know for now!

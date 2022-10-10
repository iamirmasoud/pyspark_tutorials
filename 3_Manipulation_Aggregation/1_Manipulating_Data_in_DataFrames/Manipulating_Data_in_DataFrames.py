#!/usr/bin/env python
# coding: utf-8

# # Manipulating Data in DataFrames
#
# In this lecture we will learn how to manipulate data in dataframes. You will need these techniques to accomplish some of the following tasks:
#
#  - Change data types when they are incorrectly interpretted
#  - Clean your data
#  - Create new columns
#  - Rename columns
#  - Extract or Create New Values
#
# We will also cover how to manipulate arrays in this lecture as well.
#
# #### So let's get started!
#
# First we will create our spark instance as we need to do at the start of every project.

# In[3]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Manipulate").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# # Spark's Immutability
#
# Before we get started, let's first take a moment to discuss the concept of Sparks Immutability. Spark DataFrames are immutable. What does that mean? Let's take a look at an example.

# In[4]:


names = spark.createDataFrame([("Abraham", "Lincoln")], ["first_name", "last_name"])
print(names.show())
print(names.rdd.id())


# Note the dataframe id
#
# Now add a column to the dataframe and keep calling it the same name.

# In[5]:


# add a col
from pyspark.sql.functions import *

names = names.select(
    names.first_name,
    names.last_name,
    concat_ws(" ", names.first_name, names.last_name).alias("full_name"),
)


# And see how the id of the dataframe changes but the name of the dataframe is still the same. So you can go back and reload the old id if you want.

# In[6]:


print(names.show())
print(names.rdd.id())


# ## Read in our Data Science Jobs DataFrame

# In[17]:


path = "Datasets/"
videos = spark.read.csv(path + "youtubevideos.csv", inferSchema=True, header=True)


# ## About this dataset
#
# This dataset includes several months of data on daily trending YouTube videos.
#
# **Source:** https://www.kaggle.com/datasnaek/youtube-new#USvideos.csv

# Let's check out the dataframe.

# In[8]:


videos.limit(4).toPandas()


# And of course the schema.

# In[9]:


print(videos.printSchema())


# Right away we can see that a few of the variable types were not correctly infered. This is one of the common issues that you will run into in PySpark so in this lecture, we will learn how to correct it after the read in as opposed to during the read in.
#
# ## Manipulate Data
#
# Being able to manipulate data is one of the most important skills for a data scientist to have whether you are doing machine learning or even simple reporting anlaytics. So will go over all the essentials here that you will need to do just that!
#
# *Note: Before you manipulate data, if you just want to test some code, you can use the .show() or .limit(6).toPandas() method as I've shown below. This will only display the results and NOT change your dataframe and also saves on computation time. *
#
#
# ### Changing data types after read in
# First up, we see how to change data types. Many times, you will notice that Spark's "handy" infer schema is not quite so handy. So you'll end up needing to edit the data types after you read in a dataframe quite often. Here's how you do that.
#
# #### Available types:
#     - DataType
#     - NullType
#     - StringType
#     - BinaryType
#     - BooleanType
#     - DateType
#     - TimestampType
#     - DecimalType
#     - DoubleType
#     - FloatType
#     - ByteType
#     - IntegerType
#     - LongType
#     - ShortType
#     - ArrayType
#     - MapType
#     - StructField
#     - StructType
#
# ### Adding new columns to a dataframe or overwriting them
#
# You can use PySpark's .withColumn() method for both of these usecases.
#
#     Example (add new col): df.withColumn("double_age",age*2)
#     Example (overwrite existing col): df.withColumn("age",age*2)
#
# Where the first value you pass in is the name of the new column and the second calls on the existing df column name you want to call on. You don't necessarily need to manipulate the variable here either.

# In[38]:


# Notice all vars are stings above....
# let's change that
from pyspark.sql.functions import *
from pyspark.sql.types import *  # IntegerType

df = (
    videos.withColumn("views", videos["views"].cast(IntegerType()))
    .withColumn("likes", videos["likes"].cast(IntegerType()))
    .withColumn("dislikes", videos["dislikes"].cast(IntegerType()))
    .withColumn("trending_date", to_date(videos.trending_date, "dd.mm.yy"))
)  #         .withColumn("publish_time", to_timestamp(videos.publish_time, 'yyyy-MM-dd HH:mm:ss:ms'))
print(df.printSchema())
df.limit(4).toPandas()


# **Renaming Columns**
#
# If you simply needed to rename a column you use also use this method.

# In[50]:


# Simple Rename
renamed = df.withColumnRenamed("channel_title", "channel_title_new")
renamed.limit(4).toPandas()


# **Clean Data**
#
# Alright so we see that the publish_time variable could not be converted to a timestamp becuase it has those strange "T" and "Z" values between the date and the time. We essentially need to replace the "T" value with a space, and the Z value with nothing. There are a couple of ways we can do this, the first is regex which is short for regular expressions.

# **Regex**
#
# Regex is used to replace or extract all substrings of the specified string value that match regexp with repetition.
#
# The syntax here is: regexp_replace(*str, pattern, replacement*)
#
# Regex is NOT super intuitive, so if you need a refresher on regex calls visit:
#  - https://www.whoishostingthis.com/resources/regex/
#  - https://docs.oracle.com/cd/B19306_01/server.102/b14200/ap_posix001.htm#BABJDBHB

# In[39]:


from pyspark.sql.functions import regexp_extract, regexp_replace

# import pyspark.sql.functions as f

df = df.withColumn("publish_time_2", regexp_replace(df.publish_time, "T", " "))
df = df.withColumn("publish_time_2", regexp_replace(df.publish_time_2, "Z", ""))
df = df.withColumn(
    "publish_time_3", to_timestamp(df.publish_time_2, "yyyy-MM-dd HH:mm:ss.SSS")
)
print(df.printSchema())
df.select("publish_time", "publish_time_2", "publish_time_3").show(5, False)
# Notice the .000 on the end of publish_time_new as opposed to publish_time_new_t


# **Translate Function**
#
# You could also use the Translate function here to do this, where the first set of values is what you are looking for and the second set is what you want to replace those values with respectively.

# In[40]:


import pyspark.sql.functions as f

df.select(
    "publish_time",
    f.translate(f.col("publish_time"), "TZ", " ").alias("translate_func"),
).show(5, False)


# **Trim**
#
# One common function you've probably seen in almost any data processing tool including excel is the "trim" function which removes leading and trailing white space from a cell in various ways. Let's go ahead and do that with the title field.

# In[41]:


# Trim
# pyspark.sql.functions.trim(col) - Trim the spaces from both ends for the specified string column.
from pyspark.sql.functions import *

df = df.withColumn("title", trim(df.title))  # or rtrim/ltrim
df.select("title").show(5, False)

# trim_ex = spark.createDataFrame([(' 2015-04-08 ',' 2015-05-10 ')], ['d1', 'd2']) # create a dataframe - notice the extra whitespaces in the date strings
# trim_ex.show()
# print("left trim")
# trim_ex.select('d1', ltrim(trim_ex.d1)).show()
# print("right trim")
# trim_ex.select('d1', rtrim(trim_ex.d1)).show()
# print("trim")
# trim_ex.select('d1', trim(trim_ex.d1)).show()


# **Lower**
#
# Another common data cleaning technique is lower casing all values in a string. Here's how we could do that..

# In[42]:


df = df.withColumn("title", lower(df.title))  # or rtrim/ltrim
df.select("title").show(5, False)


# **Case when**
#
# We can also use the classic sql "case when" clause to recode values. Let's say we wanted to create a categorical variable that told if the video had more likes than dislikes and visa versa.

# In[49]:


print("Option#1: select or withColumn() using when-otherwise")
from pyspark.sql.functions import when

df.select(
    "likes",
    "dislikes",
    (
        when(df.likes > df.dislikes, "Good")
        .when(df.likes < df.dislikes, "Bad")
        .otherwise("Undetermined")
    ).alias("Favorability"),
).show(3)

print("Option2: select or withColumn() using expr function")
from pyspark.sql.functions import expr

df.select(
    "likes",
    "dislikes",
    expr(
        "CASE WHEN likes > dislikes THEN  'Good' WHEN likes < dislikes THEN 'Bad' ELSE 'Undetermined' END AS Favorability"
    ),
).show(3)

print("Option 3: selectExpr() using SQL equivalent CASE expression")
df.selectExpr(
    "likes",
    "dislikes",
    "CASE WHEN likes > dislikes THEN  'Good' WHEN likes < dislikes THEN 'Bad' ELSE 'Undetermined' END AS Favorability",
).show(3)


# **Concatenate**
#
# If you want to combine two variables together (given a separator) you can use the concatenate method. Let's say we wanted to combined all the text description variables of the videos here for a robust NLP exercise of some sort and we needed to have all the text in one colum to do that like this.
#
#     concat_ws(sep, *cols)

# In[54]:


from pyspark.sql.functions import concat_ws

df.select(concat_ws(" ", df.title, df.channel_title, df.tags).alias("text")).show(
    1, False
)


# **Extracting data from Date and Timestamp variables**
#
# If you have the need to extract say the year or month from a date field, you can use PySpark's SQL function library like this.
#
# Note with this analysis we stumbled apon a date conversion descrepancy here. I'll leave fixing that for a hw problem!

# In[55]:


from pyspark.sql.functions import month, year

# Other options: dayofmonth, dayofweek, dayofyear, weekofyear
df.select("trending_date", year("trending_date"), month("trending_date")).show(5)


# **Calculate the Difference between two dates**
#
# If you want to calculate the time difference between two dates, you could use PySparks datediff function which returns the number of days from start to end.
#
#     datediff(end, start)

# In[58]:


from pyspark.sql.functions import datediff

df.select(
    "trending_date",
    "publish_time_3",
    (datediff(df.trending_date, df.publish_time_3) / 365).alias("diff"),
).show(5)


# **Split a string around a pattern**
#
# If you ever need to split a string on a pattern (where the pattern is a regex), you could use PySparks split function. You could actually use this for tokenizing text which is an NLP function that we'll get into later.
#
#     df.select(split(str, pattern))
#
# *Note that this will create an array*

# In[64]:


# Split a string around pattern (pattern is a regular expression).
from pyspark.sql.functions import split

df.select("title").show(1, False)
df.select(split(df.title, " ").alias("new")).show(1, False)


# **Working with Arrays**
#
#     df.select(array_contains(df.variable, "marriage"))
#
# *note this is only available in pyspark 2.4+*
#
#
#  - .array(*cols)   -   Creates a new array column.
#  - .array_contains(col, value)  - Collection function: returns null if the array is null, true if the array contains the given value, and false otherwise.
#  - .array_distinct(col) - Collection function: removes duplicate values from the array. :param col: name of column or expression
#  - .array_except(col1, col2) - Collection function: returns an array of the elements in col1 but not in col2, without duplicates.
#  - .array_intersect(col1, col2) - Collection function: returns an array of the elements in the intersection of col1 and col2, without duplicates.
#  - .array_join(col, delimiter, null_replacement=None) - Concatenates the elements of column using the delimiter. Null values are replaced with null_replacement if set, otherwise they are ignored.
#  - .array_max(col) - Collection function: returns the maximum value of the array.
#  - .array_min(col) - Collection function: returns the minimum value of the array.
#  - .array_position(col, value) - Collection function: Locates the position of the first occurrence of the given value in the given array. Returns null if either of the arguments are null.
#  - .array_remove(col, element)- Collection function: Remove all elements that equal to element from the given array.
#  - .array_repeat(col, count) - Collection function: creates an array containing a column repeated count times.
#  - .array_sort(col) - Collection function: sorts the input array in ascending order. The elements of the input array must be orderable. Null elements will be placed at the end of the returned array.
#  - .array_union(col1, col2) - Collection function: returns an array of the elements in the union of col1 and col2, without duplicates.
#  - .arrays_overlap(a1, a2) - Collection function: returns true if the arrays contain any common non-null element; if not, returns null if both the arrays are non-empty and any of them contains a null element; returns false otherwise.
#  - .arrays_zip(*cols)[source] - Collection function: Returns a merged array of structs in which the N-th struct contains all N-th values of input arrays.
#
#

# In[69]:


from pyspark.sql.functions import *

array_df = df.select("title", split(df.title, " ").alias("title_array"))

array_df.select("title", array_contains(array_df.title_array, "marriage")).show(
    1, False
)

# get rid of repeat values
array_df.select(array_distinct(array_df.title_array)).show(1, False)

# Remove certian values
array_df.select(array_remove(array_df.title_array, "we")).show(1, False)


# ## Creating Functions
#
# Functions as you know them in Python work a bit differently in Pyspark because it operates on a cluster. If you define a function the traditional Python way in PySpark, you will not recieve an error message but the call will not distribute on all nodes. So it will run slower.
#
# So to convert a Python function to what's called a user defined function (UDF) in PySpark. This is what you do.
#
# *Note: keep in mind that a function will not work on a column with null values

# In[70]:


from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


def square(x):
    return int(x**2)


square_udf = udf(lambda z: square(z), IntegerType())

df.select("dislikes", square_udf("dislikes").alias("likes_sq")).where(
    col("dislikes").isNotNull()
).show()


# ## That's all folks!

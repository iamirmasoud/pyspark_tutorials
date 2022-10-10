#!/usr/bin/env python
# coding: utf-8

# # Pyspark Dataframe Basics Master
#
# ## Spark DataFrame Basics
#
# Spark DataFrames are the workhouse and main way of working with Spark and Python post Spark 2.0. DataFrames act as powerful versions of tables, with rows and columns, easily handling large datasets. The shift to DataFrames provides many advantages:
# * A much simpler syntax
# * Ability to use SQL directly in the dataframe
# * Operations are automatically distributed across RDDs
#
# If you've used R or even the pandas library with Python you are probably already familiar with the concept of DataFrames. Spark DataFrame expand on a lot of these concepts, allowing you to transfer that knowledge easily by understanding the simple syntax of Spark DataFrames. Remember that the main advantage to using Spark DataFrames vs those other programs is that Spark can handle data across many RDDs, huge data sets that would never fit on a single computer. That comes at a slight cost of some "peculiar" syntax choices, but after this course you will feel very comfortable with all those topics!

# ## What is PySpark?
# **PySpark** is the collaboration of **Apache Spark** and **Python**.
#
# Apache Spark is an open-source cluster-computing framework, built around speed, ease of use, and streaming analytics whereas Python is a general-purpose, high-level programming language. It provides a wide range of libraries and is majorly used for Machine Learning and Real-Time Streaming Analytics.
#
# In other words, it **is a Python API for Spark** that lets you harness the simplicity of Python and the power of Apache Spark in order to tame Big Data and perform massive distributed processing over resilient sets of data. It's a must for Big data’s lovers.
#
# ## How is PySpark different than Python?
#
# One of the most noteable differences you will find with PySpark as opposed to Python is that it runs on a SparkContext which is a cluster, so certian processes will look different especially when you get in the machine learning libraries. In addition to this main difference, I've note a few attibutes to be aware of below:
#
# 1. PySpark does not use indexing
# 2. **ALL** objects in PySpark are **immutable**
# 3. Error messages are much less informative
# 4. Many of the libraries you are used to using in Python won't function in PySpark
#
# ## Contents of this notebook
# This notebook is intended to provide students with a easily searchable repository of all the content covered in the Dataframe Essentials portion of the course. I hope you all find this documentation useful!
#
# ## Some helpful additional resources
#
# - Exploring S3 Keys:https://alexwlchan.net/2017/07/listing-s3-keys/
# - Using S3 Select: https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-s3select.html
# - PySpark Cheat Sheets:
#     https://www.qubole.com/resources/pyspark-cheatsheet/
#     https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf
#
# ## Spark UI
# If you're ever curious about how your PySpark instance is performing, Spark offers a neat Web UI with tons of information. Just navigate to http://[driver]:4040 in your browswer where "drive" is you driver name. If you are running PySpark locally, it would be http://localhost:4040
#
#
# ## Let's Get started!
#
# Starting a PySpark Session

# In[2]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Review2").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
appid = spark._jsc.sc().applicationId()
print("You are working with", cores, "core(s) on appid: ", appid)
spark


# ## Reading in data

# In[2]:


path = "Datasets/"

# CSV
df = spark.read.csv(path + "students.csv", inferSchema=True, header=True)

# Json
people = spark.read.json(path + "people.json")

# Parquet
parquet = spark.read.parquet(path + "users.parquet")

# Partioned Parquet
partitioned = spark.read.parquet(path + "users*")

# Parts of a partitioned Parquet
users1_2 = spark.read.option("basePath", path).parquet(
    path + "users1.parquet", path + "users2.parquet"
)


# ### Notice the type differences here

# In[ ]:


type(df)


# In[ ]:


df2 = df.toPandas()
type(df2)


# ## Validate Schema and content at a glance
#
# Always a good idea to do this to ensure that dataframe was read in correctly.

# In[5]:


# Get an inital view of your dataframe
df.show(3)


# In[6]:


# and the schema
df.printSchema()


# In[ ]:


# Or just the data types which you can call on as a list
df.dtypes


# In[ ]:


# Or just the column names
df.columns


# In[ ]:


# Or just the type of one column
df.schema["Class/ASD Traits "].dataType


# In[7]:


# If your dataframe is more than just a few variables, this method is way better
df.limit(5).toPandas()


# In[8]:


# Neat "describe" function
df.describe(["gpa"]).show()


# In[9]:


# Summary function
df.select("gpa", "grade").summary("count", "min", "25%", "75%", "max").show()


# ## Specify data types as you read in datasets.
#
# Some data types make it easier to infer schema (like tabular formats such as csv which we will show later).
#
# However you often have to set the schema yourself if you aren't dealing with a .read method that doesn't have inferSchema() built-in.
#
# Spark has all the tools you need for this, it just requires a very specific structure.
#
# I've also included Spark's link to their latest list of data types for your reference in case you need it: https://spark.apache.org/docs/latest/sql-reference.html

# In[ ]:


from pyspark.sql.types import *  # StructField,StringType,IntegerType,StructType

# In[ ]:


data_schema = [
    StructField("age", IntegerType(), True),
    StructField("name", StringType(), True),
]
final_struc = StructType(fields=data_schema)
people = spark.read.json(path + "people.json", schema=final_struc)


# ## Writing Data
#
# CSV

# In[ ]:


# Note the funky naming convention of the file in your output folder. There is no way to directly change this.
df.write.mode("overwrite").csv("write_test.csv")


# **Parquet files**

# In[ ]:


df.write.mode("overwrite").parquet("parquet/")


# For those who got an error attempting to run the above code. Try this solution: https://stackoverflow.com/questions/59220832/unable-to-write-spark-dataframe-to-a-parquet-file-format-to-c-drive-in-pyspark
#
# #### Writting Partitioned Parquet Files
#
# Best practice

# In[ ]:


df.write.mode("overwrite").partitionBy("grade").parquet("partitioned_parquet/")


# #### Writting your own dataframes in Jupyter Notebooks!
#
# You can also create your own dataframes directly here in your Juypter Notebook too if you want.
#
# Like this!

# In[ ]:


values = [
    ("Pear", 10),
    ("Orange", 36),
    ("Banana", 123),
    ("Kiwi", 48),
    ("Peach", 16),
    ("Strawberry", 1),
]
df = spark.createDataFrame(values, ["fruit", "quantity"])
df.show()


# ## Select Data

# In[3]:


from pyspark.sql.functions import *

# **Basic Select**

# In[13]:


df.select(["first", "last"]).show(5)


# **Order By**

# In[11]:


df.select(["first", "last", "gpa"]).orderBy("gpa").show(5)


# **Order By Descending**

# In[14]:


df.select(["first", "last", "gpa"]).orderBy(df["gpa"].desc()).show(5)


# **Like**

# In[1]:


df.select("first", "last").where(df.first.like("%I%")).show(5, False)


# **Substrings**

# In[6]:


df.select("last", df.last.substr(1, 3)).show(5, False)


# **IS IN**

# In[2]:


df[df.first.isin("Imran", "Lu")].limit(4).toPandas()


# **Starts with Ends with**
#
# Search for a specific case - begins with "x" and ends with "x"

# In[3]:


df.select("first", "last").where(df.first.startswith("I")).where(
    df.first.endswith("n")
).limit(4).toPandas()


# **Slicing**
#
# pyspark.sql.functions.slice(x, start, length)[source] <br>
# Returns an array containing all the elements in x from index start (or starting from the end if start is negative) with the specified length.  <br>
# <br>
# *Note: indexing starts at 1 here*

# In[6]:


from pyspark.sql.functions import slice

df = spark.createDataFrame([([1, 2, 3],), ([4, 5],)], ["x"])
df.show()
df.select(
    slice(df.x, 2, 2).alias("sliced")
).show()  # first number is starting index (index starts with 1), second number is for how many


# If you want to just slice your dataframe you can do this....

# In[ ]:


# Starting
print("Starting row cound:", df.count())
print("Starting column count:", len(df.columns))

# Slice rows
df2 = df.limit(300)
print("Sliced row count:", df2.count())

# Slice columns
cols_list = df.columns[0:5]
df3 = df.select(cols_list)
print("Sliced column count:", len(df3.columns))


# ## Filtering Data
#
# A large part of working with DataFrames is the ability to quickly filter out data based on conditions. Spark DataFrames are built on top of the Spark SQL platform, which means that is you already know SQL, you can quickly and easily grab that data using SQL commands, or using the DataFram methods (which is what we focus on in this course).

# In[ ]:


fifa.filter("Overall>50").limit(4).toPandas()


# In[ ]:


# Using SQL with .select()
fifa.filter("Overall>50").select(["ID", "Name", "Nationality", "Overall"]).limit(
    4
).toPandas()


# ### Collecting Results as Python Objects

# In[ ]:


# Collecting results as Python objects
# you need the ".collect()" call at the end to "collect" the results
result = (
    fifa.select(["Nationality", "Name", "Age", "Overall"])
    .filter("Overall>70")
    .orderBy(fifa["Overall"].desc())
    .collect()
)


# In[ ]:


print("Best Player Over 70: ", result[0][1])


# Rows can also be called to turn into dictionaries if needed

# In[ ]:


row.asDict()


# In[ ]:


for item in result[0]:
    print(item)


# ## SQL Options in Spark
#
# ### Spark SQL
#
# Spark TempView provides two functions that allow users to run SQL queries against a Spark DataFrame:
#
# createOrReplaceTempView: The lifetime of this temporary view is tied to the [[SparkSession]] that was used to create this Dataset. It creates (or replaces if that view name already exists) a lazily evaluated "view" that you can then use like a hive table in Spark SQL. It does not persist to memory unless you cache the dataset that underpins the view.
#
# createGlobalTempView: The lifetime of this temporary view is tied to this Spark application.

# In[ ]:


# Create a temporary view of the dataframe
df.createOrReplaceTempView("tempview")


# In[ ]:


# Then Query the temp view
spark.sql("SELECT * FROM tempview WHERE Count > 1000").limit(5).toPandas()


# In[ ]:


# Or pass it to an object
sql_results = spark.sql(
    "SELECT * FROM tempview WHERE Count > 1000 AND Region='South West'"
)
sql_results.limit(5).toPandas()


# In[ ]:


spark.sql("SELECT Region, sum(Count) AS Total FROM tempview GROUP BY Region").limit(
    5
).toPandas()


# ### SQL Transformer
#
# You also have the option to use the SQL transformer option where you can write freeform SQL scripts.

# In[ ]:


# First we need to import SQL transformer
from pyspark.ml.feature import SQLTransformer

# In[ ]:


sqlTrans = SQLTransformer(statement="SELECT PFA,Region,Offence FROM __THIS__")
sqlTrans.transform(df).show(5)


# In[ ]:


sqlTrans = SQLTransformer(
    statement="SELECT Offence, SUM(Count) as Total FROM __THIS__ GROUP BY Offence"
)
sqlTrans.transform(df).show(5)


# # GroupBy and Aggregate Functions
#
# Let's learn how to use GroupBy and Aggregate methods on a DataFrame. GroupBy allows you to group rows together based off some column value, for example, you could group together sales data by the day the sale occured, or group repeast customer data based off the name of the customer. Once you've performed the GroupBy operation you can use an aggregate function off that data. An aggregate function aggregates multiple rows of data into a single output, such as taking the sum of inputs, or counting the number of inputs.
#
# Let's see some examples on an example dataset!

# In[ ]:


# Groupby Function with count (you can also use sum, min, max)
df.groupBy("neighbourhood_group").count().show(100)


# In[ ]:


# Then you can add the following aggregate functions: mean, count, min, max, sum
# Like this for example
df.groupBy("neighbourhood_group").mean("price").show()


# In[ ]:


# This is also a pretty neat function you can use:
summary = df.summary("count", "min", "25%", "75%", "max")
summary.toPandas()

# or a prettier version
limit_summary = df.select(
    "price",
    "minimum_nights",
    "number_of_reviews",
    "last_review",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
).summary("count", "min", "max")
limit_summary.toPandas()


# In[ ]:


# Here's another way of doing it
df.select(countDistinct("neighbourhood_group"), avg("price"), stddev("price")).show()


# **Aggregate on the entire DataFrame without groups (shorthand for df.groupBy.agg()).**

# In[ ]:


# Aggregate!
# agg(*exprs)
# Aggregate on the entire DataFrame without groups (shorthand for df.groupBy.agg()).
# available agg functions: min, max, count, countDistinct, approx_count_distinct
# df.agg.(covar_pop(col1, col2)) Returns a new Column for the population covariance of col1 and col2
# df.agg.(covar_samp(col1, col2)) Returns a new Column for the sample covariance of col1 and col2.
# df.agg(corr(col1, col2)) Returns a new Column for the Pearson Correlation Coefficient for col1 and col2.
from pyspark.sql import functions as F

df.agg(F.min(df.price).alias("Min Price")).show()


# In[ ]:


# Max sales across everything
df.agg({"number_of_reviews": "max"}).withColumnRenamed(
    "max(number_of_reviews)", "Max Reviews"
).show()


# In[ ]:


# And then if you want to group by you can do this:
df.groupBy("neighbourhood").agg({"number_of_reviews": "max"}).show()


# **Pivot Function**
#
# Provides a two way table

# In[ ]:


# Pivot Function
# pivot(pivot_col, values=None)
df.filter("room_type='Shared room'").groupBy("room_type").pivot(
    "neighbourhood_group", ["Queens", "Brooklyn"]
).count().show(100)


# ## Joining and Appending DataFrames in PySpark

# In[ ]:


valuesA = [
    ("Pirate", 1, "Arrrg"),
    ("Monkey", 2, "Oooo"),
    ("Ninja", 3, "Yaaaa"),
    ("Spaghetti", 4, "Slurp!"),
]
TableA = spark.createDataFrame(valuesA, ["name", "id", "sound"])

valuesB = [
    ("Rutabaga", 1, 2),
    ("Pirate", 2, 45),
    ("Ninja", 3, 102),
    ("Darth Vader", 4, 87),
]
TableB = spark.createDataFrame(valuesB, ["name", "id", "age"])

print("This is TableA")
print(TableA.show())
print("And this is TableB")
print(TableB.show())


# ## Appends
#
# Appending "appends" two dataframes together that have the exact same variables. You can think of it like stacking two or more blocks ON TOP of each other. To demonstrate this, we will simply join the same dataframe to itself since we don't really have a use case for this with our courses database. But hopefully this will help you imagine what to do.
#
# A common usecase would be joining the same table of infomation from one year to another year (i.e. 2012 + 2013 + ...)

# In[ ]:


new_df = TableA
df_concat = TableA.union(new_df)
print(("TableA df Counts:", TableA.count(), len(c)))
print(("df_concat Counts:", df_concat.count(), len(df_concat.columns)))
print(TableA.show(5))
print(df_concat.show(5))


# ## Joins!
#
# All options:

# In[ ]:


inner_join = TableA.join(TableB, ["name", "id"], "inner")
print("Inner Join Example")
print(inner_join.show())

left_join = TableA.join(
    TableB, ["name", "id"], how="left"
)  # Could also use 'left_outer'
print("Left Join Example")
print(left_join.show())

conditional_join = TableA.join(TableB, ["name", "id"], how="left").filter(
    TableB.name.isNull()
)
print("Conditional Left Join")
print(conditional_join.show())

right_join = TableA.join(
    TableB, ["name", "id"], how="right"
)  # Could also use 'right_outer'
print("Right Join")
print(right_join.show())

full_outer_join = TableA.join(
    TableB, ["name", "id"], how="full"
)  # Could also use 'full_outer'
print("Full Outer Join")
print(full_outer_join.show())


# ## Handling Missing Data

# In[ ]:


from pyspark.sql import functions as F

# In[ ]:


df.filter(df.cuisines.isNull()).select(["name", "cuisines"]).show()


# **Missing Data Statistics**

# In[ ]:


from pyspark.sql.functions import col, count, isnan, when

nulls = df.select(
    [count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
)
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


# In[ ]:


from pyspark.sql.functions import *


def null_value_count(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if nullRows > 0:
            temp = k, nullRows
            null_columns_counts.append(temp)
    return null_columns_counts


null_columns_count_list = null_value_count(df)
spark.createDataFrame(
    null_columns_count_list, ["Column_With_Null_Value", "Null_Values_Count"]
).show()


# **Drop all missing data**
#
# PySpark has a really handy .na function for working with missing data. The drop command has the following parameters:
#
#     df.na.drop(how='any', thresh=None, subset=None)

# In[ ]:


df.na.drop().limit(4).toPandas()


# In[ ]:


# Of course you will want to know how many rows that affected before you actually execute it..
og_len = df.count()
drop_len = df.na.drop().count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[ ]:


# Drop rows that have at least 8 NON-null values
og_len = df.count()
drop_len = df.na.drop(thresh=8).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[ ]:


# Only drop the rows whose values in the sales column are null
og_len = df.count()
drop_len = df.na.drop(subset=["rate"]).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[ ]:


# Another way to do the above
og_len = df.count()
drop_len = df.filter(zomato.rate.isNotNull()).count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# In[ ]:


# Drop a row only if ALL its values are null.
og_len = df.count()
drop_len = df.na.drop(how="all").count()
print("Total Rows Dropped:", og_len - drop_len)
print("Percentage of Rows Dropped", (og_len - drop_len) / og_len)


# ### Fill the missing values
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


# A very common practice is to fill values with the mean value for the column. Here is a fun function to that in an automatted way.

# In[ ]:


def fill_with_mean(df, include=set()):
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    #     stats = stats.select(*(col(c).cast("int").alias(c) for c in stats.columns)) #IntegerType()
    return df.na.fill(stats.first().asDict())


updated_df = fill_with_mean(df, ["approx_cost(for two people)", "votes"])
updated_df.limit(5).toPandas()


# ## Manipulating Data in DataFrames
#
# Change data types
#
# ### Available types:
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

# In[ ]:


from pyspark.sql.functions import *
from pyspark.sql.types import *

df = (
    videos.withColumn("views", videos["views"].cast(IntegerType()))
    .withColumn("likes", videos["likes"].cast(IntegerType()))
    .withColumn("dislikes", videos["dislikes"].cast(IntegerType()))
    .withColumn("trending_date", to_date(videos.trending_date, "dd.mm.yy"))
)
#         .withColumn("publish_time", to_timestamp(videos.publish_time, 'yyyy-MM-dd HH:mm:ss:ms'))
print(df.printSchema())
df.limit(4).toPandas()


# **Regex**
#
# Regex is used to replace or extract all substrings of the specified string value that match regexp with rep.
# regexp_replace(str, pattern, replacement)
# for more info on regex calls visit: https://docs.oracle.com/cd/B19306_01/server.102/b14200/ap_posix001.htm#BABJDBHB

# In[ ]:


import pyspark.sql.functions as f
from pyspark.sql.functions import regexp_extract, regexp_replace

df = df.withColumn("publish_time", regexp_replace(df.publish_time, "T", " "))
df = df.withColumn("publish_time", regexp_replace(df.publish_time, "Z", ""))
df = df.withColumn(
    "publish_time", to_timestamp(df.publish_time, "yyyy-MM-dd HH:mm:ss.SSS")
)
print(df.printSchema())
df.limit(4).toPandas()


# **Translate Function**

# In[ ]:


# You can also use the translate function for cases like this
# where you wanted to replace ('$', '#', ',') with ('X', 'Y', 'Z')
import pyspark.sql.functions as f

foobar = spark.createDataFrame(
    [("$100,00",), ("#foobar",), ("foo, bar, #, and $",)], ["A"]
)
foobar.select("A", f.translate(f.col("A"), "$#,", "XYZ").alias("replaced")).show()


# **Trim**

# In[ ]:


# Trim
# pyspark.sql.functions.trim(col) - Trim the spaces from both ends for the specified string column.
from pyspark.sql.functions import *

trim_ex = spark.createDataFrame(
    [(" 2015-04-08 ", " 2015-05-10 ")], ["d1", "d2"]
)  # create a dataframe - notice the extra whitespaces in the date strings
trim_ex.show()
print("left trim")
trim_ex.select("d1", ltrim(trim_ex.d1)).show()
print("right trim")
trim_ex.select("d1", rtrim(trim_ex.d1)).show()
print("trim")
trim_ex.select("d1", trim(trim_ex.d1)).show()


# **Case When**

# In[ ]:


df = spark.createDataFrame([(1, 1), (2, 2), (3, 3)], ["id", "value"])

print("Sample Dataframe:")
df.show()

print("Option#1: withColumn() using when-otherwise")
from pyspark.sql.functions import when

df.withColumn(
    "value_desc",
    when(df.value == 1, "one").when(df.value == 2, "two").otherwise("other"),
).show()

print("Option2: withColumn() using expr function")
from pyspark.sql.functions import expr

df.withColumn(
    "value_desc",
    expr(
        "CASE WHEN value == 1 THEN  'one' WHEN value == 2 THEN  'two' ELSE 'other' END AS value_desc"
    ),
).show()

print("Option 3: selectExpr() using SQL equivalent CASE expression")
fifa.selectExpr(
    "*",
    "CASE WHEN value == 1 THEN  'one' WHEN value == 2 THEN  'two' ELSE 'other' END AS value_desc",
).show()

print("Option 4: select() using expr function")
from pyspark.sql.functions import expr

df.select(
    "*",
    expr(
        "CASE WHEN value == 1 THEN  'one' WHEN value == 2 THEN  'two' ELSE 'other' END AS value_desc"
    ),
).show()


# **Creating new columns calculated using existing columns**

# In[ ]:


# Add a new column from an existing column like this....
# withColumn(colName, col)[source]
# Returns a new DataFrame by adding a column or replacing the existing column that has the same name.
# The column expression must be an expression over this DataFrame; attempting to add a column from some other dataframe will raise an error.

# Parameters
# colName – string, name of the new column.

# col – a Column expression for the new column.
views = df.withColumn("views_x_2", df.views * 2)
views.select(["views", "views_x_2"]).show(4)


# In[ ]:


# You can also use this method to overwrite a column
views = views.withColumn("views", views.views * 2)
views.select(["views", "views_x_2"]).show(4)


# **Renaming Columns**

# In[ ]:


# Simple Rename
renamed = df.withColumnRenamed("channel_title", "channel_title_new")
renamed.limit(4).toPandas()


# **Concatenate**

# In[ ]:


from pyspark.sql.types import *  # IntegerType

# Concatenate columns
# pyspark.sql.functions.concat_ws(sep, *cols)[source]
# Concatenates multiple input string columns together into a single string column, using the given separator.

names = spark.createDataFrame([("Abraham", "Lincoln")], ["first_name", "last_name"])
names.select(
    names.first_name,
    names.last_name,
    concat_ws(" ", names.first_name, names.last_name).alias("full_name"),
).show()


# **Extracting from Date and Timestamp variables**

# In[ ]:


# Extract year, month, day etc. from a date field
# Other options: dayofmonth, dayofweek, dayofyear, weekofyear
import pyspark.sql.functions as fn

year = df.withColumn("TRENDING_YEAR", fn.year("trending_date")).withColumn(
    "TRENDING_MONTH", fn.month("trending_date")
)
# QA
year.filter("TRENDING_YEAR=2011").select(
    ["trending_date", "TRENDING_YEAR", "TRENDING_MONTH"]
).show()


# In[ ]:


# Calculate the difference between two dates:
# pyspark.sql.functions.datediff(end, start)
# Returns the number of days from start to end.

date_df = spark.createDataFrame([("2015-04-08", "2015-05-10")], ["d1", "d2"])
date_df.select(datediff(date_df.d2, date_df.d1).alias("diff")).show()


# **Splitting a string around a pattern**

# In[ ]:


# Split a string around pattern (pattern is a regular expression).
from pyspark.sql.functions import *

# pyspark.sql.functions.split(str, pattern)[source]

abc = spark.createDataFrame(
    [("ab12cd",)],
    [
        "s",
    ],
)
abc.select(abc.s, split(abc.s, "[0-9]+").alias("news")).show()


# **Arrays**
#
# *Note that the array_distinct feature is new in Spark 2.4.

# In[ ]:


# Arrays - col/cols – list of column names (string) or list of Column expressions that have the same data type.
# pyspark.sql.functions
# note this is only available in pyspark 2.4+
from pyspark.sql.functions import *

#      .array(*cols)   -   Creates a new array column.
#      .array_contains(col, value)  - Collection function: returns null if the array is null, true if the array contains the given value, and false otherwise.
#      .array_distinct(col) - Collection function: removes duplicate values from the array. :param col: name of column or expression
#      .array_except(col1, col2) - Collection function: returns an array of the elements in col1 but not in col2, without duplicates.
#      .array_intersect(col1, col2) - Collection function: returns an array of the elements in the intersection of col1 and col2, without duplicates.
#      .array_join(col, delimiter, null_replacement=None) - Concatenates the elements of column using the delimiter. Null values are replaced with null_replacement if set, otherwise they are ignored.
#      .array_max(col) - Collection function: returns the maximum value of the array.
#      .array_min(col) - Collection function: returns the minimum value of the array.
#      .array_position(col, value) - Collection function: Locates the position of the first occurrence of the given value in the given array. Returns null if either of the arguments are null.
#      .array_remove(col, element)- Collection function: Remove all elements that equal to element from the given array.
#      .array_repeat(col, count) - Collection function: creates an array containing a column repeated count times.
#      .array_sort(col) - Collection function: sorts the input array in ascending order. The elements of the input array must be orderable. Null elements will be placed at the end of the returned array.
#      .array_union(col1, col2) - Collection function: returns an array of the elements in the union of col1 and col2, without duplicates.
#      .arrays_overlap(a1, a2) - Collection function: returns true if the arrays contain any common non-null element; if not, returns null if both the arrays are non-empty and any of them contains a null element; returns false otherwise.
#      .arrays_zip(*cols)[source] - Collection function: Returns a merged array of structs in which the N-th struct contains all N-th values of input arrays.

customer = spark.createDataFrame(
    [("coffee", "milk", "coffee", "coffee", "chocolate", "")],
    ["item1", "item2", "item3", "item4", "item5", "item6"],
)
purchases = customer.select(
    array("item1", "item2", "item3").alias("Monday"),
    array("item4", "item5", "item6").alias("Tuesday"),
)

print("array")
purchases.show()

print("Which customers purchased milk? array_contains")
purchases.select(array_contains(purchases.Monday, "milk")).show(1, False)

print("List of unique products purchased on Monday: array_distinct")
purchases.select(array_distinct(purchases.Monday)).show(1, False)

print("What did our customers order on Monday but not Tuesday? array_except")
purchases.select(array_except(purchases.Monday, purchases.Tuesday)).show(1, False)

print("What did our customers order on BOTH Monday and Tuesday?: array_intersect")
purchases.select(array_intersect(purchases.Monday, purchases.Tuesday)).show(1, False)

print("All purchases on monday in a string: array_join")
purchases.select(array_join(purchases.Monday, ",")).show(1, False)


# **Create an array by splitting a string field**

# In[ ]:


from pyspark.sql.functions import *

values = [
    (45, "I like to ride bikes"),
    (14, "I like chicken"),
    (63, "I like bubbles"),
    (75, "I like roller coasters"),
    (24, "I like shuffle board"),
    (45, "I like to swim"),
]
sentences = spark.createDataFrame(values, ["age", "sentence"])
df = sentences.withColumn("array", split(col("sentence"), " "))
df.show(1, False)


# ## Creating Functions
#
# Functions as you know them in Python work a bit differently in Pyspark because it operates on a cluster. If you define a function the traditional Python way in PySpark, you will not recieve an error message but the call will not distribute on all nodes. So it will run slower.
#
# So to convert a Python function to what's called a user defined function (UDF) in PySpark. This is what you do.

# In[ ]:


from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType


def square(x):
    return int(x**2)


square_udf = udf(lambda z: square(z), IntegerType())

df.select("age", square_udf("age").alias("age_squared")).show()

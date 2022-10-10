#!/usr/bin/env python
# coding: utf-8

# # SQL Options in Spark
#
# PySpark provides two main options when it comes to using staight SQL. Spark SQL and SQL Transformer.
#
# ## 1. Spark SQL
#
# Spark TempView provides two functions that allow users to run **SQL** queries against a Spark DataFrame:
#
#  - **createOrReplaceTempView:** The lifetime of this temporary view is tied to the SparkSession that was used to create the dataset. It creates (or replaces if that view name already exists) a lazily evaluated "view" that you can then use like a hive table in Spark SQL. It does not persist to memory unless you cache the dataset that underpins the view.
#  - **createGlobalTempView:** The lifetime of this temporary view is tied to this Spark application. This feature is useful when you want to share data among different sessions and keep alive until your application ends.
#
# A **Spark Session vs. Spark application:**
#
# **Spark application** can be used:
#
# - for a single batch job
# - an interactive session with multiple jobs
# - a long-lived server continually satisfying requests
# - A Spark job can consist of more than just a single map and reduce.
# - can consist of more than one Spark Session.
#
# A **SparkSession** on the other hand:
#
#  - is an interaction between two or more entities.
#  - can be created without creating SparkConf, SparkContext or SQLContext, (they’re encapsulated within the SparkSession which is new to Spark 2.0)
#
#
# ## 2. SQL Transformer
#
# You also have the option to use the SQL transformer option where you can write free-form SQL scripts as well.
#
# # SQL Options within regular PySpark calls
#
# 1. The expr function in PySparks SQL Function Library
# 2. PySparks selectExpr function
#
# We will go over all these in detail so buckel up!
#
#
# Let's start with Spark SQL. But first we need to create a Spark Session!

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Let's Read in our DataFrame for this Notebook
#
# ### About this data
#
# Recorded crime for the Police Force Areas of England and Wales. The data are rolling 12-month totals, with points at the end of each financial year between year ending March 2003 to March 2007 and at the end of each quarter from June 2007.
#
# **Source:** https://www.kaggle.com/r3w0p4/recorded-crime-data-at-police-force-area-level

# In[2]:


# Start by reading a basic csv dataset
# Let Spark know about the header and infer the Schema types!

path = "Datasets/"

crime = spark.read.csv(path + "rec-crime-pfa.csv", header=True, inferSchema=True)


# In[16]:


# This is way better
crime.limit(5).toPandas()


# In[17]:


print(crime.printSchema())


# So, in order for us to perform SQL calls off of this dataframe, we will need to rename any variables that have spaces in them. We will not be using the first variable so I'll leave that one as is, but we will be using the last variable, so I will go ahead and change that to Count so we can work with it.

# In[4]:


df = crime.withColumnRenamed(
    "Rolling year total number of offences", "Count"
)  # .withColumn("12 months ending", crime["12 months ending"].cast(DateType())).
print(df.printSchema())


# In[5]:


# Create a temporary view of the dataframe
df.createOrReplaceTempView("tempview")


# In[11]:


# Then Query the temp view
spark.sql("SELECT * FROM tempview WHERE Count > 1000").limit(5).toPandas()


# In[9]:


# Or choose which vars you want
spark.sql("SELECT Region, PFA FROM tempview WHERE Count > 1000").limit(5).toPandas()


# In[21]:


# You can also pass your query results to an object
# (we don't need to use .collect() here)
sql_results = spark.sql(
    "SELECT * FROM tempview WHERE Count > 1000 AND Region='South West'"
)
sql_results.limit(5).toPandas()


# In[10]:


# We can even do aggregated "group by" calls like this
spark.sql("SELECT Region, sum(Count) AS Total FROM tempview GROUP BY Region").limit(
    5
).toPandas()


# basically anything goes

# ### SQL Transformer
#
# You also have the option to use the SQL transformer option where you can write freeform SQL scripts.

# In[7]:


# First we need to import SQL transformer
from pyspark.ml.feature import SQLTransformer

# In[10]:


# Then we create an SQL call
sqlTrans = SQLTransformer(statement="SELECT PFA,Region,Offence FROM __THIS__")
# And use it to transform our df object
sqlTrans.transform(df).show(5)


# In[28]:


type(sqlTrans)


# In[25]:


# Note that "__THIS__" is a special word and cannot be change to __THAT__ for example
sqlTrans = SQLTransformer(statement="SELECT PFA,Region,Offence FROM __THAT__")
# And use it to transform our df object
sqlTrans.transform(df).show(5)


# In[23]:


# Also Note that a call like this won't work...
SQLTransformer(statement="SELECT PFA,Region,Offence FROM __THIS__").show()


# **Now how about a group by call**

# In[26]:


# Note that this call will not work on the original dataframe "crime" when the variable "Count" is a string

sqlTrans = SQLTransformer(
    statement="SELECT Offence, SUM(Count) as Total FROM __THIS__ GROUP BY Offence"
)
sqlTrans.transform(df).show(5)


# **And a where statement**

# In[27]:


sqlTrans = SQLTransformer(
    statement="SELECT PFA,Offence FROM __THIS__ WHERE Count > 1000"
)
sqlTrans.transform(df).show(5)


# **You can also, of course, read the output into a dataframe**

# In[29]:


result = sqlTrans.transform(df)
result.show(5)


# # SQL Options within regular PySpark calls
#
# ### The expr function in PySparks SQL Function Library
#
# You can also use the expr function within the pyspark.sql.functions library coupled with either PySpark's withColumn function or the select function.

# In[30]:


# First we need to read in the library
from pyspark.sql.functions import expr

# Let's add a percent column to the dataframe. To do this, first we need to get the total number of rows in the dataframe (we can't soft this unfortunatly).

# In[34]:


sqlTrans = SQLTransformer(statement="SELECT SUM(Count) as Total FROM __THIS__")
sqlTrans.transform(df).show(5)


# In[36]:


# We could add a percent column to our df
# that shows the offence %
# with the "withColumn" command
df.withColumn("percent", expr("round((count/244720928)*100,2)")).show()


# In[35]:


# Same thing with the "select" command
df.select("*", expr("round((count/244720928)*100,2) AS percent")).show()


# ### PySparks selectExpr function
#
# Very similar idea here but slightly different syntax.

# In[37]:


df.selectExpr("*", "round((count/244720928)*100,2) AS percent").filter(
    "Region ='South West'"
).show()


# ## That's all folks! Great job!

# In[ ]:


# Speed test


# In[15]:


spark.sql("SELECT * FROM tempview WHERE Count > 1000").show()


# In[ ]:


# Then we create an SQL call
sqlTrans = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE Count > 1000")
# And use it to transform our df object
sqlTrans.transform(df).show(5)


# In[16]:


# Then we create an SQL call
SQLTransformer(statement="SELECT * FROM __THIS__ WHERE Count > 1000").transform(
    df
).show()


# In[ ]:

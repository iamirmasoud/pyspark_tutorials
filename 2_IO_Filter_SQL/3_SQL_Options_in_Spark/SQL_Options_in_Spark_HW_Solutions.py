#!/usr/bin/env python
# coding: utf-8

# # SQL Options in Spark HW Solutions
#
# Alirght let's apply what we learned in the lecture to a new dataset!
#
# **But first!**
#
# Let's start with Spark SQL. But first we need to create a Spark Session!

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("SparkSQLHWSolutions").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in our DataFrame for this Notebook
#
# For this notebook we will be using the Google Play Store csv file attached to this lecture. Let's go ahead and read it in.
#
# ### About this dataset
#
# Contains a list of Google Play Store Apps and info about the apps like the category, rating, reviews, size, etc.
#
# **Source:** https://www.kaggle.com/lava18/google-play-store-apps

# In[3]:


path = "Datasets/"

googlep = spark.read.csv(path + "googleplaystore.csv", header=True, inferSchema=True)


# ## First things first
#
# Let's check out the first few lines of the dataframe to see what we are working with

# In[4]:


# This is way better
googlep.limit(5).toPandas()


# As well as the schema to make sure all the column types were correctly infered

# In[5]:


print(googlep.printSchema())


# Looks like we need to edit some of the datatypes. Let's just update Rating, Reviews and Price as integer (float for Rating) values for now, since the Size and Installs variables will need a bit more cleaning.

# In[6]:


from pyspark.sql.types import FloatType, IntegerType

df = (
    googlep.withColumn("Rating", googlep["Rating"].cast(FloatType()))
    .withColumn("Reviews", googlep["Reviews"].cast(IntegerType()))
    .withColumn("Price", googlep["Price"].cast(IntegerType()))
)
print(df.printSchema())
df.limit(5).toPandas()


# Looks like that worked! Great! Let's dig in.

# ## 1. Create Tempview
#
# Go ahead and create a tempview of the dataframe so we can work with it in spark sql.

# In[7]:


# Create a temporary view of the dataframe
df.createOrReplaceTempView("tempview")


# ## 2. Select all apps with ratings above 4.1
#
# Use your tempview to select all apps with ratings above 4.1

# In[8]:


# Then Query the temp view
spark.sql("SELECT * FROM tempview WHERE Rating > 4.1").limit(5).toPandas()


# ## 3. Now pass your results to an object (ie create a spark dataframe)
#
# Select just the App and Rating column where the Category is in the Comic category and the Rating is above 4.5.

# In[9]:


# Or pass it to an object
sql_results = spark.sql(
    "SELECT App,Rating FROM tempview WHERE Category = 'COMICS' AND Rating > 4.5"
)
sql_results.limit(5).toPandas()


# ## 4. Which category has the most cumulative reviews
#
# Only select the one category with the most reivews.
#
# *Note: will require adding all the review together for each category*

# In[10]:


spark.sql(
    "SELECT Category, sum(Reviews) AS Total_Reviews FROM tempview GROUP BY Category ORDER BY Total_Reviews DESC"
).limit(1).toPandas()


# ## 5. Which App has the most reviews?
#
# Display ONLY the top result
#
# Include only the App column and the Reviews column.

# In[11]:


spark.sql("SELECT App, Reviews FROM tempview ORDER BY Reviews DESC").show(1)


# ## 5. Select all apps that contain the word 'dating' anywhere in the title
#
# *Note: we did not cover this in the lecture. You'll have to use your SQL knowledge :) Google it if you need to.*

# In[12]:


spark.sql("SELECT * FROM tempview WHERE App LIKE '%dating%'").limit(5).toPandas()


# ## 6. Use SQL Transformer to display how many free apps there are in this list

# In[13]:


# First we need to import SQL transformer
from pyspark.ml.feature import SQLTransformer

# In[14]:


sqlTrans = SQLTransformer(statement="SELECT count(*) FROM __THIS__ WHERE Type = 'Free'")
sqlTrans.transform(df).show()


# ## 7. What is the most popular Genre?
#
# Which genre appears most often in the dataframe. Show only the top result.

# In[15]:


sqlTrans = SQLTransformer(
    statement="SELECT Genres, count(*) as Total FROM __THIS__ GROUP BY Genres ORDER BY Total DESC"
)
sqlTrans.transform(df).show(1)


# ## 8. Select all the apps in the 'Tools' genre that have more than 100 reviews

# In[16]:


sqlTrans = SQLTransformer(
    statement="SELECT App, Reviews FROM __THIS__ WHERE Genres = 'Tools' AND Reviews > 100"
)
sqlTrans.transform(df).show(10)


# ## That's all folks! Great job!

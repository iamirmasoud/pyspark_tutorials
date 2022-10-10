#!/usr/bin/env python
# coding: utf-8

# # Aggregating DataFrames in PySpark HW Solutions
#
# First let's start up our PySpark instance

# In[2]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("aggregate").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in the dataFrame for this Notebook

# In[3]:


airbnb = spark.read.csv("Datasets/nyc_air_bnb.csv", inferSchema=True, header=True)


# ## About this dataset
#
# This dataset describes the listing activity and metrics for Air BNB bookers in NYC, NY for 2019. Each line in the dataset is a booking.
#
# **Source:** https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/data
#
# Let's go ahead and view the first few records of the dataset so we know what we are working with.

# In[3]:


airbnb.limit(5).toPandas()


# In[4]:


print(airbnb.printSchema())


# Notice here that some of the columns that are obviously numeric have been incorrectly identified as "strings". Let's edit that. Otherwise we cannot aggregate any of the numeric columns.

# In[4]:


from pyspark.sql.functions import *
from pyspark.sql.types import *

df = (
    airbnb.withColumn("price", airbnb["price"].cast(IntegerType()))
    .withColumn("minimum_nights", airbnb["minimum_nights"].cast(IntegerType()))
    .withColumn("number_of_reviews", airbnb["number_of_reviews"].cast(IntegerType()))
    .withColumn("reviews_per_month", airbnb["reviews_per_month"].cast(IntegerType()))
    .withColumn(
        "calculated_host_listings_count",
        airbnb["calculated_host_listings_count"].cast(IntegerType()),
    )
)
# QA
print(df.printSchema())
df.limit(5).toPandas()


# ### Alright now we are ready to dig in!
#
#
# ### 1. How many rows are in this dataset?

# In[5]:


df.count()


# ### 2. How many total reviews does each host have?

# In[6]:


df.groupBy("host_id").sum("number_of_reviews").show(10)


# ### 3. Show the min and max of all the numeric variables in the dataset

# In[11]:


limit_summary = df.select(
    "price",
    "minimum_nights",
    "number_of_reviews",
    "last_review",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
).summary("min", "max")
limit_summary.toPandas()


# ### 4. Which host had the highest number of reviews?
#
# Only display the top result.
#
# Bonus: format the column names

# In[12]:


from pyspark.sql import functions

df.groupBy("host_id").agg(sum("number_of_reviews").alias("Reviews")).orderBy(
    sum("number_of_reviews").desc()
).show(1)


# ### 5. On average, how many nights did most hosts specify for a minimum?

# In[23]:


df.agg({"minimum_nights": "avg"}).withColumnRenamed(
    "avg(minimum_nights)", "Avg Min Nights"
).show()


# In[10]:


df.agg(mean(df.minimum_nights)).show()


# ### 6. What is the most expensive neighborhood to stay in on average?
#
# Note: only show the one result

# In[13]:


result = df.groupBy("neighbourhood").agg(avg(df.price).alias("avg_price"))
result.orderBy(result.avg_price.desc()).show(1)


# ### 7. Display a two by two table that shows the average prices by room type (private and shared only) and neighborhood group (Manhattan and Brooklyn only)

# In[11]:


df.filter("room_type IN('Private room','Shared room')").groupBy("room_type").pivot(
    "neighbourhood_group", ["Manhattan", "Brooklyn"]
).avg("price").show(100)


# ### Alright that's all folks!
#
# ### Great job!

# In[ ]:

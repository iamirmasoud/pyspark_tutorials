#!/usr/bin/env python
# coding: utf-8

# # Aggregating DataFrames in PySpark
#
# In this lecture we will be going over how to aggregate dataframes in Pyspark.
# The commands we will learn here will be super useful for doing quality checks
# on your dataframes and answering more simiplistic business questions with you data.
#
# So let's get to it! Here is what we will cover today:
#
#  - GroupBy
#  - Pivot
#  - Aggregate methods
#  - Combos of each

# In[1]:

# Run this bit for Windows users
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

# In[2]:


# Start by reading in a basic csv dataset
# Let Spark know about the header and infer the Schema types!

# Some csv data
airbnb = spark.read.csv("Datasets/nyc_air_bnb.csv", inferSchema=True, header=True)


# ## About this dataset
#
# This dataset describes the listing activity and metrics for Air BNB bookers in NYC, NY for 2019. Each line in the dataset is a booking.
#
# **Source:** https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/data
#
# Let's go ahead and view the first view lines of the dataframe.

# In[3]:


airbnb.limit(5).toPandas()


# In[4]:


print(airbnb.printSchema())


# Notice here that some of the columns that are obviously numeric have been incorrectly identified as "strings". Let's edit that. Otherwise we cannot aggregate any of the numeric columns.

# In[5]:


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


# # GroupBy and Aggregate Functions
#
# Let's learn how to use GroupBy and Aggregate methods on a DataFrame. These two commands go hand in hand many times in PySpark. ACtually in order to use the GroupBy command, you have to also tell Spark what numeric aggregate you want to learn about. For example, count, average or min/max.
#
# GroupBy allows you to group rows together based off some column value, for example, you could group together sales data by the day the sale occured, or group repeat customer data based off the name of the customer. Once you've performed the GroupBy operation you can use an aggregate function off that data. An aggregate function aggregates multiple rows of data into a single output, such as taking the sum of inputs, or counting the number of inputs.
#
# You can also use the aggreate function independently as well to learn about overall statistics of your dataframe too which we will see in some of our examples.
#
# So let's dig in!

# In[6]:


# For example we may be interested to see how many listings there were per neighbourhood group.
# Groupby Function with count (you can also use sum, min, max)
df.groupBy("neighbourhood_group").count().show(7)


# In[23]:


# Then you can add the following aggregate functions: mean, count, min, max, sum
# Like this for example
df.groupBy("neighbourhood_group").mean("price").show(5)


# In[21]:


# This is another way of doing the above but I don't recommend it
# because you can only do one var at a time
df.groupBy("neighbourhood").agg({"price": "mean"}).show(5)


# In[45]:


# This method is way more versatile
# Allows you to call on more than one aggregate function at a time
# It's my fav for this reason!
from pyspark.sql.functions import *

df.groupBy("neighbourhood").agg(
    min(df.price).alias("Min Price"), max(df.price).alias("Max Price")
).show(5)


# In[8]:


# This is also a pretty neat function you can use:
summary = df.summary("count", "min", "25%", "75%", "max")
summary.toPandas()
# But be careful because it'll perform this operation on your whole df!


# In[11]:


# Eh that was ugly!
# To do a summary for specific columns first select them:
# limit_summary = df.select("price","minimum_nights","number_of_reviews","last_review","reviews_per_month","calculated_host_listings_count","availability_365").summary("count","min","max")
limit_summary = df.select("price", "minimum_nights", "number_of_reviews").summary(
    "count", "min", "max"
)
limit_summary.toPandas()


# ### Aggregate on the entire DataFrame without groups (shorthand for df.groupBy.agg()).
#
# This is great, but what if we wanted the overall summary metrics like average and counts for more than one variable and without a groupBy variable? We could do this using the pyspark.sql functions library.

# In[25]:


# Aggregate!
# agg(*exprs)
# Aggregate on the entire DataFrame without groups (shorthand for df.groupBy.agg()).
# available agg functions: min, max, count, countDistinct, approx_count_distinct
# df.agg.(covar_pop(col1, col2)) Returns a new Column for the population covariance of col1 and col2
# df.agg.(covar_samp(col1, col2)) Returns a new Column for the sample covariance of col1 and col2.
# df.agg(corr(col1, col2)) Returns a new Column for the Pearson Correlation Coefficient for col1 and col2.

from pyspark.sql.functions import *

df.agg(min(df.price).alias("Min Price"), max(df.price).alias("Max Price")).show()


# In[24]:


# There is also this method which is pretty similar
df.select(
    countDistinct("neighbourhood_group").alias("CountD"), avg("price"), stddev("price")
).show()


# In[18]:


# You could also write the syntax like this....
# But keep in mind with this method that you can only do one variable at a time (bummer)
# Again I don't recommend this!
# Max sales across everything
df.agg({"number_of_reviews": "max"}).withColumnRenamed(
    "max(number_of_reviews)", "Max Reviews"
).show()


# ### Pivot Function
#
# Provides a two way table and must be used in conjunction with groupBy.

# In[35]:


# Pivot Function
# pivot(pivot_col, values=None)
df.groupBy("room_type").pivot(
    "neighbourhood_group", ["Queens", "Brooklyn"]
).count().show(10)


# In[ ]:


# You can also filter your results if you need to
# We some invalid data in the above output
# So we could select only the "Share room" types if we wanted to
df.filter("room_type='Shared room'").groupBy("room_type").pivot(
    "neighbourhood_group", ["Queens", "Brooklyn"]
).count().show(100)


# ### Comine all three!
#
# It is also possible to combine all three method into one call: GroupBy, Pivot and Agg like this:

# In[44]:


# from pyspark.sql.functions import *
df.groupBy("neighbourhood").pivot("neighbourhood_group", ["Queens", "Brooklyn"]).agg(
    min(df.price).alias("Min Price"), max(df.price).alias("Max Price")
).toPandas()  # .show()
# Note The toPandas() method should only be used if the resulting Pandas’s DataFrame is expected to be small,
# as all the data is loaded into the driver’s memory.


# In[ ]:

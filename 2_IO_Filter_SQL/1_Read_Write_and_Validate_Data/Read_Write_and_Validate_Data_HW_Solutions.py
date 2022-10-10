#!/usr/bin/env python
# coding: utf-8

# # Reading, Writing and Validating Data in PySpark HW Solutions
#
# Welcome to your first coding homework assignment in PySpark! I hope you enjoyed the lecture on Reading, Writing and Validating dataframes. Now it's time to put what you've learned into action!
#
# I've included several instructions below to help guide you through this homework assignment which I hope will get you feeling even comfortable reading, writing and validating dataframes. If you get at any point, feel free to jump to the next lecture where I will guide you through my solutions to the HW assignment.
#
# Have fun!
#
# Let's dig right in!
#
#
# ## But first things first.....
# We need to always begin every Spark session by creating a Spark instance. Let's go ahead and use the method we learned in the lecture in the cell below. Also see if you can remember how to open the Spark UI (using a link that automatically guides you there).

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("ReadWriteVal").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Next let's start by reading a basic csv dataset
#
# Download the pga_tour_data_historical dataset that is attached to this lecture and save it whatever folder you want, then read it in.
#
# **Data Source:** https://www.kaggle.com/bradklassen/pga-tour-20102018-data
#
# Rememer to try letting Spark infer the header and infer the Schema types!

# In[2]:


path = "Datasets/"

# Some csv data
pga = spark.read.csv(path + "pga_tour_historical.csv", inferSchema=True, header=True)


# ## 1. View first 5 lines of dataframe
# First generate a view of the first 5 lines of the dataframe to get an idea of what is inside. We went over two ways of doing this... see if you can remember BOTH ways.

# In[3]:


pga.show(3)


# In[4]:


# I prefer this method
pga.limit(5).toPandas()


# ## 2. Print the schema details
#
# Now print the details of the dataframes schema that Spark infered to ensure that it was infered correctly. Sometimes it is not infered correctly, so we need to watch out!

# In[5]:


print(pga.printSchema())
print("")
print(pga.columns)
print("")
# Not so fond of this method, but to each their own
print(pga.describe())


# ## 3. Edit the schema during the read in
#
# We can see from the output above that Spark did not correctly infer that the "value" column was an integer value. Let's try specifying the schema this time to let spark know what the schema should be.
#
# Here is a link to see a list of PySpark data types in case you need it (also attached to the lecture): https://spark.apache.org/docs/latest/sql-reference.html#data-types

# In[6]:


from pyspark.sql.types import IntegerType, StringType, StructField, StructType

# In[7]:


data_schema = [
    StructField("Player Name", StringType(), True),
    StructField("Season", IntegerType(), True),
    StructField("Statistic", StringType(), True),
    StructField("Variable", StringType(), True),
    StructField("Value", IntegerType(), True),
]


# In[8]:


final_struc = StructType(fields=data_schema)


# In[9]:


path = "Datasets/"
pga = spark.read.csv(path + "pga_tour_historical.csv", schema=final_struc)


# In[10]:


pga.printSchema()
# That's better!


# ## 4. Generate summary statistics for only one variable
#
# See if you can generate summary statistics for only the "Value" column using the .describe function
#
# (count, mean, stddev, min, max)

# In[11]:


# Neat "describe" function
pga.describe(["Value"]).show()


# ## 5. Generate summary statistics for TWO variables
# Now try to generate ONLY the count min and max for BOTH the "Value" and "Season" variable using the select. You can't use the .describe function for this one but see if you can remember which function you CAN use.

# In[12]:


pga.select("Season", "Value").summary("count", "min", "max").show()


# ## 6. Write a parquet file
#
# Now try writing a parquet file (not partitioned) from the pga dataset. But first create a new dataframe containing ONLY the the "Season" and "Value" fields (using the "select command you used in the question above) and write a parquet file partitioned by "Season". This is a bit of a challenge aimed at getting you ready for material that will be covered later on in the course. Don't feel bad if you can't figure it out.
#
# *Note that if any of your variable names contain spaces, spark will produce an error message with this call. That is why we are selecting ONLY the "Season" and "Value" fields. Ideally we should renamed those columns but we haven't gotten to that yet in this course but we will soon!*

# In[14]:


df = pga.select("Season", "Value")
df.write.mode("overwrite").parquet("partition_parquet/")


# ## 7. Write a partioned parquet file
#
# You will need to use the same limited dataframe that you created in the previous question to accomplish this task as well. Use the variable "Season" as you partitioning variable here.

# In[15]:


df.write.mode("overwrite").partitionBy("Season").parquet("partitioned_parquet/")
df.show(5)


# ## 8. Read in a partitioned parquet file
#
# Now try reading in the partitioned parquet file you just created above.

# In[16]:


path = "partitioned_parquet/"  # Note: if you add a * to the end of the path, the Season var will be automatically dropped
parquet = spark.read.parquet(path)

parquet.show()


# ## 9. Reading in a set of paritioned parquet files
#
# Now try only reading Seasons 2010, 2011 and 2012.

# In[17]:


# Notice that this method only gives you the "Value" column
path = "partitioned_parquet/"
partitioned = spark.read.parquet(
    path + "Season=2010/", path + "Season=2011/", path + "Season=2012/"
)

partitioned.show(5)


# In[18]:


# We need to use this method to get the "Season" and "Value" Columns
path = "partitioned_parquet/"
dataframe = spark.read.option("basePath", path).parquet(
    path + "Season=2010/", path + "Season=2011/", path + "Season=2012/"
)
dataframe.show(5)


# ## 10. Create your own dataframe
#
# Try creating your own dataframe below using PySparks *.createDataFrame* function. See if you can make one that contains 4 variables and at least 3 rows.
#
# Let's see how creative you can get on the content of the dataframe :)

# In[19]:


values = [
    ("Kyle", 10, "A", 1),
    ("Melbourne", 36, "A", 1),
    ("Nina", 123, "A", 1),
    ("Stephen", 48, "B", 2),
    ("Orphan", 16, "B", 2),
    ("Imran", 1, "B", 2),
]
df = spark.createDataFrame(values, ["name", "age", "AB", "Number"])
df.show()


# ## We're done! Great job!

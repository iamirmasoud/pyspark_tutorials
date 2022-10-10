#!/usr/bin/env python
# coding: utf-8

# # Frequent Pattern Mining in PySpark's MLlib Project Solution
# 
# Let's see if you can use the concepts we learned about in the lecture to try out frequent pattern mining techniques on a new dataset!
# 
# 
# ## Recap:
# 
# Spark MLlib implements two algorithms related to frequency pattern mining (FPM): 
# 
# - FP-growth
# - PrefixSpan 
# 
# The distinction is that FP-growth does not use order information in the itemsets, if any, while PrefixSpan is designed for sequential pattern mining where the itemsets are ordered. 
# 
# ## Data
# 
# You are owing a supermarket mall and through membership cards, you have some basic data about your customers like Customer ID, age, gender, annual income and spending score. Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.
# 
# ## Problem statement
# 
# You own the mall and want to understand the customers like who can be easily grouped together so that a strategy can be provided to the marketing team to plan accordingly.
# 
# **Source:**  https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
# May take awhile locally
spark = SparkSession.builder.appName("FPM_Project").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark
# Click the hyperlinked "Spark UI" link to view details about your Spark session


# **Read in the dataframe**

# In[2]:


path ="Datasets/"
df = spark.read.csv(path+'Mall_Customers.csv',inferSchema=True,header=True)


# In[3]:


df.limit(4).toPandas()


# In[4]:


df.printSchema()


# In[5]:


# Let's rename some of these column names to be a bit more user friendly
# Sometime Spark will not be able to process a command if the var names have spaces or special characters
df = df.withColumnRenamed("Annual Income (k$)", "income")
df = df.withColumnRenamed("Spending Score (1-100)", "spending_score")
df.show(5)


# In[6]:


#How many rows do we have in our dataframe?
df.count()


# ## Create a meaningful grouping system
# 
# We need to recode our values so they can be grouped and analyzed accordingly. Let's do that here.

# In[8]:


from pyspark.sql.functions import *

groups = df.withColumn("age_group",expr("CASE WHEN Age < 30 THEN 'Under 30' WHEN Age BETWEEN 30 AND 55 THEN '30 to 55' WHEN Age > 50 THEN '50 +' ELSE 'Other' END AS age_group"))
print(groups.groupBy("age_group").count().show())

groups = groups.withColumn("income_group",expr("CASE WHEN income < 40 THEN 'Under 40' WHEN income BETWEEN 40 AND 70 THEN '40 - 70' WHEN income > 70 THEN '70 +' ELSE 'Other' END AS income_group"))
print(groups.groupBy("income_group").count().show())

groups = groups.withColumn("spending_group",expr("CASE WHEN spending_score < 30 THEN 'Less than 30' WHEN spending_score BETWEEN 30 AND 60 THEN '30 - 60' WHEN spending_score > 60 THEN '60 +' ELSE 'Other' END AS spending_group"))
print(groups.groupBy("spending_group").count().show())

print(groups.groupBy("Gender").count().show())

groups = groups.withColumn("items",array('Gender','age_group', 'income_group','spending_group')) #items is what spark is expecting
groups.limit(4).toPandas()


# ## Fit the FPGrowth model
# 
# Since order does not matter here. 

# In[9]:


from pyspark.ml.fpm import FPGrowth
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.1)
model = fpGrowth.fit(groups)


# ## Determine item popularity
# 
# See what combos were most popular

# In[10]:


itempopularity = model.freqItemsets
itempopularity.createOrReplaceTempView("itempopularity")
# Then Query the temp view
print("Top 20")
spark.sql("SELECT * FROM itempopularity ORDER BY freq desc").limit(200).toPandas()


# ## Review Association Rules
# 
# In addition to freqItemSets, the FP-growth model also generates **associationRules**. For example, if a shopper purchases peanut butter, what is the probability (or confidence) that they will also purchase jelly.  For more information, a good reference is Susan Li’s *A Gentle Introduction on Market Basket Analysis — Association Rules*
# 
# A good way to think about association rules is that model determines that if you purchased something (i.e. the antecedent), then you will purchase this other thing (i.e. the consequent) with the following confidence.
# 
# **Source:** https://databricks.com/blog/2018/09/18/simplify-market-basket-analysis-using-fp-growth-on-databricks.html

# In[12]:


# Display generated association rules.
assoc = model.associationRules
assoc.createOrReplaceTempView("assoc")
# Then Query the temp view
print("Top 20")
spark.sql("SELECT * FROM assoc ORDER BY confidence desc").limit(200).toPandas()


# ## Take aways
# 
# Awesome! So we see that the highest confidence group was the [40 - 70] income group paired with the [30-60] spending group which means that our advice to the marketing team might be to focus efforts on this group first. 

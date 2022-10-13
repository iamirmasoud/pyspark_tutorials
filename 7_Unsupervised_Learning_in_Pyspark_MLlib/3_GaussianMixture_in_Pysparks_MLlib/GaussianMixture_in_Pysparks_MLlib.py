#!/usr/bin/env python
# coding: utf-8

# # Gaussian Mixture Modeling in PySpark's MLlib
#
# As we discussed in the concept review lecture, GMM is a "soft" clustering method that provides a probability of how associated a data point is with a cluster as opposed to simply which cluster the data point is associated with. It works esspecially well for data that has multiple distributions, but can be used for any kind of data.
#
# If you would like to learn even more about GMM I recommend the following article: https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
#
# **Link to GMM in PySpark Documentation:**<br>
# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.GaussianMixture

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Gix").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## About this data
#
# Sample Sales Data, Order Info, Sales, Customer, Shipping, etc., Used for Segmentation, Customer Analytics, Clustering and More. Inspired for retail analytics. This was originally used for Pentaho DI Kettle, But I found the set could be useful for Sales Simulation training.
#
# Originally Written by María Carina Roldán, Pentaho Community Member, BI consultant (Assert Solutions), Argentina. This work is licensed under the Creative Commons Attribution-Noncommercial-Share Alike 3.0 Unported License. Modified by Gus Segura June 2014.
#
# **Source:** https://www.kaggle.com/kyanyoga/sample-sales-data

# In[13]:


path = "Datasets/"
df = spark.read.csv(path + "sales_data_sample.csv", inferSchema=True, header=True)


# In[9]:


df.limit(5).toPandas()


# In[10]:


df.printSchema()


# ## Import some libraries we will need
#
# You will notice that some of these are pythonic which means they won't distribute accross our dataframe, but we will only use them on situations were distribution is not usually necessary. I always try to use PySpark functions where ever I can, but sometimes the functionality I need is not available. In these cases I lean back on my trusty Python libraries :)

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *

# In[84]:


# Fill in null values with average
# Since this function passes in a df, we don't need to (and cannot) create a UDF for it
# It will distribute accross our dataframe because the functions within the function are from PySpark
def fill_with_mean(df, include=set()):
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    #     stats = stats.select(*(col(c).cast("int").alias(c) for c in stats.columns)) #IntegerType()
    return df.na.fill(stats.first().asDict())


cols_list = ["QUANTITYORDERED", "PRICEEACH", "SALES"]
df = df.select(cols_list)
columns = df.columns
df = fill_with_mean(df, columns)
df.limit(5).toPandas()


# Convert all input columns into a vector as usual

# In[15]:


input_columns = df.columns  # Collect the column names as a list
vecAssembler = VectorAssembler(inputCols=input_columns, outputCol="features")
final_df = vecAssembler.transform(df).select("features")
final_df.show()


# ## Determine optimal K

# In[32]:


kmax = 50
ll = np.zeros(kmax)
for k in range(2, kmax):
    gm = GaussianMixture(k=k, tol=0.0001, maxIter=10, seed=10)
    model = gm.fit(final_df)
    summary = model.summary
    ll[k] = summary.logLikelihood


# In[34]:


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, kmax), ll[2:kmax])
ax.set_xlabel("k")
ax.set_ylabel("ll")


# In[36]:


# for the sake of speed and simplicity we will stick with k = 5
gm = GaussianMixture(k=5, maxIter=10, seed=10)
model = gm.fit(final_df)

summary = model.summary
print("Clusters: ", summary.k)
print("Cluster Sizes: ", summary.clusterSizes)
print("Log Likelihood: ", summary.logLikelihood)

weights = model.weights
print("Model Weights: :", len(weights))

print("Means: ", model.gaussiansDF.select("mean").head())

print("Cov: ", model.gaussiansDF.select("cov").head())

transformed = model.transform(final_df)  # .select("features", "prediction")


# ## Reflection
#
# Looks the 5 clusters that our model identified range in size quite a bit. There is one cluster that only has 7 cases in it as opposed to 1,307 in the largest cluster. If we were using this dataset to target customers, we may want to focus our efforts on the largest group first as we would have more bange for our buck.

# In[49]:


transformed.limit(7).toPandas()


# Let's see if we can try to learn something about our clusters!

# In[62]:


transformed.show(1, False)


# In[81]:


transformed.groupBy("prediction").agg(
    {
        "prediction": "count",
        "QUANTITYORDERED": "min",
        "PRICEEACH": "min",
        "SALES": "min",
    }
).orderBy("prediction").show()


# In[82]:


transformed.groupBy("prediction").agg(
    {
        "prediction": "count",
        "QUANTITYORDERED": "max",
        "PRICEEACH": "max",
        "SALES": "max",
    }
).orderBy("prediction").show()


# In[83]:


transformed.groupBy("prediction").agg(
    {
        "prediction": "count",
        "QUANTITYORDERED": "mean",
        "PRICEEACH": "mean",
        "SALES": "mean",
    }
).orderBy("prediction").show()


# In[70]:


limited = transformed.filter("prediction == 0")
aggregates = limited.summary("min", "mean", "max")
print("Total Cases in this Cluster: ", limited.count())
aggregates.toPandas()


# # Conclusions
#
# Of the 5 clusters, the first one seems to buy for quality (highest price per item and max sales), while the last cluster seems to be the "buy in bulk" shoppers. Cluster 3 seems to be more of a bargain hunter (lowest price each and avg sales)

#!/usr/bin/env python
# coding: utf-8

# # K-means and Bisecting K-means in PySpark's MLlib
#
# Welcome to the first clustering code along activity!
#
# **Recall from the Intro to Clustering lecture** <br>
# Clustering is a technique for unsupervised learning which I like to think of as “Smart Grouping”. The "unsupervised" part of this type approach just means that you do not have a dependent variable to work with which means you need to get a bit more creative. That's where clustering comes in handy.
#
# ### K-means
# The k-means algorithms divides the data points of a dataset into clusters (groups) based on the nearest average (mean) values. In order to find the mean data point the algorithm tries to minimize the distance between points in each cluster.
#
# In the term "k-means", k denotes the number of clusters in the data. Since the k-means algorithm doesn’t determine this, you’re required to specify this quantity. The quality of the clusters is heavily dependent on on how good you were at guessing the value of k. If your data just has 2 or three variables to work with, that might be easy to guess by looking at a scatter plot, but when your data has several variables that approach will be nearly impossible. So we basically guess and check until we figure out the best guess.
#
# The k-means algorithm works by placing random cluster centers around your ploting area and then evaluating whether moving them in any one direction would result in a new center with better results (higher density) — with more data points closer to it.
#
# **PySpark documentation:** <br> https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.KMeans
#
# ### Bi-secting K-means
#
# Very similar to k-Means clustering. It starts with all objects in a single cluster, and then..
#
#     1. Picks a cluster to split.
#     2. Find 2 sub-clusters using the basic k-Means algorithm (Bisecting step)
#     3. Repeat step 2, the bisecting step, until there are k leaf clusters in total or no leaf clusters are divisible. The bisecting steps of clusters on the same level are grouped together to increase parallelism that Spark provides. If bisecting all divisible clusters on the bottom level would result more than k leaf clusters, larger clusters get higher priority.
#     4. Repeat steps 1, 2 and 3 until the desired number of clusters is reached.
#
#
# **PySpark documentation:** <br> https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.clustering.BisectingKMeans
#
# So let's see them both in action!

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Regression").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Import dataset
#
# **Content**
#
# This dataset summarizes the usage behavior of about 9,000 credit card holders during the last 6 months that we will use to define a marketing outreach strategy for. Basically we want to see how we can get our customers to purchase more! The file is at a customer level with 18 behavioral variables.
#
# **Data Dictionary:**
#
#  - **CUST_ID:** Identification of Credit Card holder (Categorical)
#  - **BALANCE:** Balance amount left in their account to make purchases
#  - **BALANCE_FREQUENCY:** How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
#  - **PURCHASES:** Amount of purchases made from account
#  - **ONEOFF_PURCHASES:** Maximum purchase amount done in one-go
#  - **INSTALLMENTS_PURCHASES:** Amount of purchase done in installment
#  - **CASH_ADVANCE:** Cash in advance given by the user
#  - **PURCHASES_FREQUENCY:** How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
#  - **ONEOFFPURCHASESFREQUENCY:** How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
#  - **PURCHASESINSTALLMENTSFREQUENCY:** How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
#  - **CASHADVANCEFREQUENCY:** How frequently the cash in advance being paid
#  - **CASHADVANCETRX:** Number of Transactions made with "Cash in Advanced"
#  - **PURCHASES_TRX:** Number of purchase transactions made
#  - **CREDIT_LIMIT:** Limit of Credit Card for user
#  - **PAYMENTS:** Amount of Payment done by user
#  - **MINIMUM_PAYMENTS:** Minimum amount of payments made by user
#  - **PRCFULLPAYMENT:** Percent of full payment paid by user TENURE : Tenure of credit card service for user
#
# **Source:** https://www.kaggle.com/arjunbhasin2013/ccdata

# In[2]:


path = "Datasets/"
df = spark.read.csv(path + "credit_card_data.csv", inferSchema=True, header=True)


# In[3]:


df.limit(5).toPandas()


# In[5]:


df.printSchema()


# ## Check the nulls values

# In[4]:


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


# ### Fill in null values
#
# Fill in null values with average (except for the ID column)

# In[5]:


from pyspark.sql.functions import *


def fill_with_mean(df, include=set()):
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    return df.na.fill(stats.first().asDict())


columns = df.columns
columns = columns[1:]
df = fill_with_mean(df, columns)
df.limit(5).toPandas()


# **Convert all input columns (features) into a vector**
#
# *Remember that we don't have a dependent variable here.*

# In[7]:


from pyspark.ml.feature import VectorAssembler

input_columns = df.columns  # Collect the column names as a list
input_columns = input_columns[
    1:
]  # keep only relevant columns: from column 8 until the end
vecAssembler = VectorAssembler(inputCols=input_columns, outputCol="features")
df_kmeans = vecAssembler.transform(df)  # .select('CUST_ID', 'features')
df_kmeans.limit(4).toPandas()


# ## K-means
#
# One disadvantage of KMeans compared to more advanced clustering algorithms is that the algorithm must be told how many clusters, k, it should try to find. To optimize k (find the right amount of clusters) we cluster a fraction of the data for different choices of k and look for an "elbow" in the cost function.
#
# **Source:** https://rsandstroem.github.io/sparkkmeans.html

# In[24]:


import numpy as np
# Find the best amount of clusters
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# set a max for the number of clusters you want to try out
kmax = 50
# Create and array filled with zeros for the amount of k
# Similar to creating an empty list
kmcost = np.zeros(kmax)
for k in range(2, kmax):
    # Set up the k-means alogrithm
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    # Fit it on your dataframe
    model = kmeans.fit(df_kmeans)
    # Fill in the zeros of your array with cost....
    # Computes the "cost" (sum of squared distances) between the input points and their corresponding cluster centers.
    kmcost[k] = model.computeCost(df_kmeans)  # requires Spark 2.0 or later


# Can you see the "elbow" below?

# In[25]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the plot dimensions
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Then specify the range of values for the axis and call on your cost array
ax.plot(range(2, kmax), kmcost[2:kmax])
# Set up the axis labels
ax.set_xlabel("k")
ax.set_ylabel("cost")


# Looks like the "elbow" is at about 15 or so....
#
# Less clusters is always better.

# ## Bisecting k-means
#
# Now let's compare with bi-secting k-means.

# In[26]:


from pyspark.ml.clustering import BisectingKMeans

# Same calls here except with bkmeans
kmax = 50
bkmcost = np.zeros(kmax)
for k in range(2, kmax):
    bkmeans = BisectingKMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = bkmeans.fit(df_kmeans)
    bkmcost[k] = model.computeCost(df_kmeans)  # requires Spark 2.0 or later


# In[27]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, kmax), bkmcost[2:kmax])
ax.set_xlabel("k")
ax.set_ylabel("cost")


# Looks like this line is telling a similar story but it's a bit less smooth which is concerning. This is a red flag that the model could be unreliable.
#
# #### Compare the two models
#
# 1. Plot them together
# 2. We can also compare the two models above by calculating the difference between cost values at various k values (kmeans - bisecting kmeans). So negative values means k-means is winning and positive values means bi-secting k-means is winning.
#
# Let's see who wins!

# In[41]:


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, kmax), bkmcost[2:kmax], color="blue")
ax.plot(range(2, kmax), kmcost[2:kmax], color="red")
ax.set_xlabel("k")
ax.set_ylabel("cost")


# In[31]:


def compare(bkmcost, kmcost):
    diff = []
    for k in range(2, kmax, 5):
        temp = k, (kmcost[k] - bkmcost[k])
        diff.append(temp)
    return diff


diff_list = compare(bkmcost, kmcost)
diff_list


# Looks like k-means wins by a landslide at every iteration!
#
# ## Fit final model
#
# Looks like the elbow was at around 15 so we will stick with that.
#
# Now let's train our final model and we can print out the centroids of the ten clusters.

# In[32]:


k = 15
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)

# bkmeans = BisectingKMeans().setK(k).setSeed(1).setFeaturesCol("features")
# model = bkmeans.fit(df_kmeans)

# Make predictions
predictions = model.transform(df_kmeans)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
print(" ")

# Shows the cluster centers
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[34]:


import numpy as np
# Those are a little bit hard to read
# Let's convert to a dataframe so we can add the column names
import pandas as pd

center_pdf = pd.DataFrame(list(map(np.ravel, centers)))
center_pdf.columns = columns
center_pdf


# Awesome information! We can see the centroids for each variable at each cluster. Now we can use this data to target our customer base.

# ### Check the individual predictions
#
# Let's also check out the the predictions for each row that was produced above. The prediction value is an integer between 0 and k.

# In[42]:


predictions.limit(5).toPandas()


# ## Conclusions
#
# That was awesome! Now we can use this data to target our marketing efforts! We could use these groups to target similar customer segments. For example if we do some research about the groups and discover that one is mostly a certian soicio economic status and purchasing frequency, and offer them a cost savings package that could be benficial to them. How cool would that be?!
#
# We could also learn a bit more about our clustering by calling on various aggregate statistics for each one of the clusters across each of the variables in our dataframe like this.

# In[40]:


predictions.groupBy("prediction").agg(
    min(predictions.BALANCE).alias("Min BALANCE"),
    max(predictions.BALANCE).alias("Max BALANCE"),
).show(15)


# In[ ]:

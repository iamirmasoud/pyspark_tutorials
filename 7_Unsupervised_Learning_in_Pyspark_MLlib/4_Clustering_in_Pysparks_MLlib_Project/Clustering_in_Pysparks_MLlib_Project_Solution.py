#!/usr/bin/env python
# coding: utf-8

# # Clustering in PySpark's MLlib Project Solution
#
# **Project Title:** University Clustering for the Greater Good
#
# **Objective**
#
# You are a data scientist employed by the ABCDE Foundation, a non-profit organization whose mission is to increase college graduation rates for underpriveleged populations. Through advocacy and targeted outreach programs, ABCDE strives to identify and alleviate barriers to educational achievement. ABCDE is driven by the belief that with the right supports, an increase in college attendance and completion rates can be achieved, thereby weakening the grip of the cycles of poverty and social immobility affecting many of our communities.
#
# ABCDE is committed to developing a more data driven approach to decision making. As a prelude to future analyses, ABCDE has requested that you analyze the data to identify clusters of similar colleges and universities.
#
# **Your Task**
#
# Use cluster analysis to identify the groups of characteristically similar schools in the CollegeScorecard.csv dataset.
#
# ## Considerations:
#
#  - Data Preparation
#  - How will you deal with missing values?
#  - Categorical variables?
#  - Hyperparameter optimization?
#
# These are the (sometimes subjective) questions you need to figure out as a data scientist. It's highly recommended to familiarize yourself with the dataset's dictionary and documentation, as well as the theory and technical characteristics of the algorithm(s) you're using.
#
# **Interpretation**
#
# Is it possible to explain what each cluster represents? Did you retain or prepare a set of features that enables a meaningful interpretation of the clusters? Do the compositions of the clusters seem to make sense?
#
# The CollegeScoreCardDataDictionary-09-12-2015.csv file attached will also help with this part.
#
# **Validation**
#
# How will you measure the validity of your clustering process? Which metrics will you use and how will you apply them?
#
# *Important Note*
#
# This is an open-ended assignment (as many or most real-life data science projects are). Your only constraints are that you must use the data provided, execute high-quality and justifiable clustering technique, provide your rationale for the decisions you made, and ultimately produce meaningful cluster labels.
#
# **Deliverables:**
#
# An array of cluster labels corresponding to UNITID (the unique college/university I.D. variable).
#
# **Source:** https://data.world/exercises/cluster-analysis-exercise-2

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Clustering_Project").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Import Dataset
#
# ### About this dataset
#
#

# In[2]:


path = "Datasets/"
df = spark.read.csv(path + "CollegeScorecard.csv", inferSchema=True, header=True)


# **View data**

# In[3]:


df.limit(6).toPandas()


# In[4]:


df.printSchema()


# ## Treat null values
#
# For the sake of simplicity, let's fill in all nulls values with the mean of that column. Since there are so many columns, I will just go through and treat only the numeric columns.

# In[4]:


from pyspark.sql.functions import *


def fill_with_mean(df, include=set()):
    stats = df.agg(*(avg(c).alias(c) for c in df.columns if c in include))
    #     stats = stats.select(*(col(c).cast("int").alias(c) for c in stats.columns)) #IntegerType()
    return df.na.fill(stats.first().asDict())


columns = df.columns
input_columns = []
for column in columns:
    if str(df.schema[column].dataType) in ("IntegerType", "DoubleType"):
        input_columns.append(column)

input_columns
df = fill_with_mean(df, input_columns)
df.limit(5).toPandas()


# ## Prep Data for Input
#
# Vectorize the input columns for the algorithm to process

# In[9]:


from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=input_columns, outputCol="features")
final_df = vecAssembler.transform(df).select("UNITID", "features")
final_df.show()


# ## Optimize choice of K for K means
#
# I will go with K means for this project. Let's optimize choice of K.

# In[11]:


import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

kmax = 6
cost = np.zeros(kmax)
for k in range(2, kmax):
    kmeans = (
        KMeans(initSteps=4, tol=1e-4, maxIter=20)
        .setK(k)
        .setSeed(1)
        .setFeaturesCol("features")
    )
    model = kmeans.fit(final_df.sample(False, 0.1, seed=42))
    cost[k] = model.computeCost(final_df)  # requires Spark 2.0 or later


# In[13]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, kmax), cost[2:kmax])
ax.set_xlabel("k")
ax.set_ylabel("cost")


# Looks like there was not much gain after 4. Let's go ahead with that and make our predictions.

# In[15]:


k = 4
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(final_df)

# Make predictions
predictions = model.transform(final_df)

# Evaluate cluster by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# ## Describe the clusters in lamens terms
#
# What are some defining characteristics of each cluster? Try to use some of the variables to explain to the client.

# In[19]:


# First get the centers of each cluster
centers = model.clusterCenters()

import numpy as np

# Then map the centers to their variable names
import pandas as pd

center_pdf = pd.DataFrame(list(map(np.ravel, centers)))
center_pdf.columns = input_columns
center_pdf


# In[44]:


# Now let's print some take aways for our client
print("Centroid for number of branches in each cluster:")
print(center_pdf.NUMBRANCH)
print("")
print("Centroid for Highest Degree Awarded:")
print("Pretty big disparity between clusters 0 and 1 compared to 2 and 3")
print(center_pdf.HIGHDEG)
print("")
print("Centroid for Locale (higher numbers mean more remote locale):")
print("Looks like there is not a ton of diffence in the clusters here")
print(center_pdf.LOCALE)
print("")
print("Centroid for Historically Black (binary outcome):")
print(
    "Looks like cluter one has a slightly higher rate of historically black universities"
)
print(center_pdf.HBCU)


# ## How many Universities are in each cluster?

# ### First Assign clusters to each row in the original dataframe

# In[22]:


transformed = model.transform(final_df).select("UNITID", "prediction")
rows = transformed.collect()
print(rows[:3])


# In[23]:


df_pred = spark.createDataFrame(rows)
df_pred.show(5)


# ### Join the predictions to the original dataframe so we have all the columns back in their original form

# In[25]:


results = df_pred.join(df, "UNITID")
results.limit(5).toPandas()


# In[28]:


results.groupBy("prediction").count().show()


# ## List the Universities by Cluster

# In[31]:


uni_list = results.select("INSTNM").filter("prediction == 0")
uni_list.show(truncate=False)


# In[32]:


uni_list = results.select("INSTNM").filter("prediction == 1")
uni_list.show(truncate=False)


# In[33]:


uni_list = results.select("INSTNM").filter("prediction == 2")
uni_list.show(truncate=False)


# In[ ]:

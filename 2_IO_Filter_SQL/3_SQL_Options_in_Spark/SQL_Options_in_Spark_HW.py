#!/usr/bin/env python
# coding: utf-8

# # SQL Options in Spark HW
#
# Alirght let's apply what we learned in the lecture to a new dataset!
#
# **But first!**
#
# Let's start with Spark SQL. But first we need to create a Spark Session!

# In[ ]:


# ## Read in our DataFrame for this Notebook
#
# For this notebook we will be using the Google Play Store csv file attached to this lecture. Let's go ahead and read it in.
#
# ### About this dataset
#
# Contains a list of Google Play Store Apps and info about the apps like the category, rating, reviews, size, etc.
#
# **Source:** https://www.kaggle.com/lava18/google-play-store-apps

# In[ ]:


# ## First things first
#
# Let's check out the first few lines of the dataframe to see what we are working with

# In[ ]:


# As well as the schema to make sure all the column types were correctly infered

# In[ ]:


# Looks like we need to edit some of the datatypes. We need to update Rating, Reviews and Price as integer (float for Rating) values for now, since the Size and Installs variables will need a bit more cleaning. Since we haven't been over this yet, I'm going to provide the code for you here so you can get a quick look at how it used (and how often we need it!).
#
# **make sure to change the df name to whatever you named your df**

# In[ ]:


from pyspark.sql.types import FloatType, IntegerType

newdf = (
    df.withColumn("Rating", df["Rating"].cast(FloatType()))
    .withColumn("Reviews", df["Reviews"].cast(IntegerType()))
    .withColumn("Price", df["Price"].cast(IntegerType()))
)
print(newdf.printSchema())
newdf.limit(5).toPandas()


# Looks like that worked! Great! Let's dig in.

# ## 1. Create Tempview
#
# Go ahead and create a tempview of the dataframe so we can work with it in spark sql.

# In[ ]:


# ## 2. Select all apps with ratings above 4.1
#
# Use your tempview to select all apps with ratings above 4.1

# In[ ]:


# ## 3. Now pass your results to an object
# (ie create a spark dataframe)
#
# Select just the App and Rating column where the Category is in the Comic category and the Rating is above 4.5.

# In[ ]:


# ## 4. Which category has the most cumulative reviews
#
# Only select the one category with the most reivews.
#
# *Note: will require adding all the review together for each category*

# In[ ]:


# ## 5. Which App has the most reviews?
#
# Display ONLY the top result
#
# Include only the App column and the Reviews column.

# In[ ]:


# ## 5. Select all apps that contain the word 'dating' anywhere in the title
#
# *Note: we did not cover this in the lecture. You'll have to use your SQL knowledge :) Google it if you need to.*

# In[ ]:


# ## 6. Use SQL Transformer to display how many free apps there are in this list

# In[ ]:


# ## 7. What is the most popular Genre?
#
# Which genre appears most often in the dataframe. Show only the top result.

# In[ ]:


# ## 8. Select all the apps in the 'Tools' genre that have more than 100 reviews

# In[ ]:


# ## That's all folks! Great job!

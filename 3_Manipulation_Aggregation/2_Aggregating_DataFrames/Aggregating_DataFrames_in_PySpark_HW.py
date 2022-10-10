#!/usr/bin/env python
# coding: utf-8

# # Aggregating DataFrames in PySpark HW
#
# First let's start up our PySpark instance

# In[ ]:


# ## Read in the dataFrame for this Notebook

# In[2]:


airbnb = spark.read.csv("Datasets/nyc_air_bnb.csv", inferSchema=True, header=True)


# ## About this dataset
#
# This dataset describes the listing activity and metrics for Air BNB bookers in NYC, NY for 2019. Each line in the dataset is a booking.
#
# **Source:** https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/data
#
# Let's go ahead and view the first few records of the dataset so we know what we are working with.

# In[ ]:


# Now print the schema so we can make sure all the variables have the correct types

# In[ ]:


# Notice here that some of the columns that are obviously numeric have been incorrectly identified as "strings". Let's edit that. Otherwise we cannot aggregate any of the numeric columns.

# In[ ]:


# ### Alright now we are ready to dig in!
#
#
# ### 1. How many rows are in this dataset?

# In[ ]:


# ### 2. How many total reviews does each host have?

# In[ ]:


# ### 3. Show the min and max of all the numeric variables in the dataset

# In[ ]:


# ### 4. Which host had the highest number of reviews?
#
# Only display the top result.
#
# Bonus: format the column names

# In[ ]:


# ### 5. On average, how many nights did most hosts specify for a minimum?

# In[ ]:


# ### 6. What is the most expensive neighborhood to stay in on average?
#
# Note: only show the one result

# In[ ]:


# ### 7. Display a two by two table that shows the average prices by room type (private and shared only) and neighborhood group (Manhattan and Brooklyn only)

# In[ ]:


# ### Alright that's all folks!
#
# ### Great job!

# In[ ]:

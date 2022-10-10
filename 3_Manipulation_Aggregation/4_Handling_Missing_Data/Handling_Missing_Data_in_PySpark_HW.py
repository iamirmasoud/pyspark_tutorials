#!/usr/bin/env python
# coding: utf-8

# # Handling Missing Data in PySpark HW Solutions
#
# In this HW assignment you will be strengthening your skill sets dealing with missing data.
#
# **Review:** you have 2 basic options for filling in missing data (you will personally have to make the decision for what is the right approach:
#
# 1. Drop the missing data points (including the entire row)
# 2. Fill them in with some other value.
#
# Let's practice some examples of each of these methods!
#
#
# #### But first!
#
# Start your Spark session

# In[ ]:


# ## Read in the dataset for this Notebook
#
# Weather.csv attached to this lecture.

# In[ ]:


# ## About this dataset
#
# **New York City Taxi Trip - Hourly Weather Data**
#
# Here is some detailed weather data for the New York City Taxi Trips.
#
# **Source:** https://www.kaggle.com/meinertsen/new-york-city-taxi-trip-hourly-weather-data

# ### Print a view of the first several lines of the dataframe to see what our data looks like

# In[ ]:


# ### Print the schema
#
# So that we can see if we need to make any corrections to the data types.

# In[ ]:


# ## 1. How much missing data are we working with?
#
# Get a count and percentage of each variable in the dataset to answer this question.

# In[ ]:


# ## 2. How many rows contain at least one null value?
#
# We want to know, if we use the df.na option, how many rows will we loose.

# In[ ]:


# ## 3. Drop the missing data
#
# Drop any row that contains missing data across the whole dataset

# In[ ]:


# ## 4. Drop with a threshold
#
# Count how many rows would be dropped if we only dropped rows that had a least 12 NON-Null values

# In[ ]:


# ## 5. Drop rows according to specific column value
#
# Now count how many rows would be dropped if you only drop rows whose values in the tempm column are null/NaN

# In[ ]:


# ## 6. Drop rows that are null accross all columns
#
# Count how many rows would be dropped if you only dropped rows where ALL the values are null

# In[ ]:


# ## 7. Fill in all the string columns missing values with the word "N/A"
#
# Make sure you don't edit the df dataframe itself. Create a copy of the df then edit that one.

# In[ ]:


# ## 8. Fill in NaN values with averages for the tempm and tempi columns
#
# *Note: you will first need to compute the averages for each column and then fill in with the corresponding value.*

# In[ ]:


# ### That's it! Great Job!

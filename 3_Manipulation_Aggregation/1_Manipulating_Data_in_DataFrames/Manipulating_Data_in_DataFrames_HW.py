#!/usr/bin/env python
# coding: utf-8

# # Manipulating Data in DataFrames HW
#
#
# #### Let's get started applying what we learned in the lecure!
#
# I've provided several questions below to help test and expand you knowledge from the code along lecture. So let's see what you've got!
#
# First create your spark instance as we need to do at the start of every project.

# In[ ]:


# ## Read in our Republican vs. Democrats Tweet DataFrame
#
# Attached to the lecture

# In[ ]:


# ## About this dataframe
#
# Extracted tweets from all of the representatives (latest 200 as of May 17th 2018)
#
# **Source:** https://www.kaggle.com/kapastor/democratvsrepublicantweets#ExtractedTweets.csv
#
# Use either .show() or .toPandas() check out the first view rows of the dataframe to get an idea of what we are working with.

# In[ ]:


# **Prevent Truncation of view**
#
# If the view you produced above truncated some of the longer tweets, see if you can prevent that so you can read the whole tweet.

# In[ ]:


# **Print Schema**
#
# First, check the schema to make sure the datatypes are accurate.

# In[ ]:


# ## 1. Can you identify any tweet that mentions the handle @LatinoLeader using regexp_extract?
#
# It doesn't matter how you identify the row, any identifier will do. You can test your script on row 5 from this dataset. That row contains @LatinoLeader.

# In[ ]:


# ## 2. Replace any value other than 'Democrate' or 'Republican' with 'Other' in the Party column.
#
# We can see from the output below, that there are several other values other than 'Democrate' or 'Republican' in the Part column. We are assuming that this is dirty data that needs to be cleaned up.

# In[ ]:


# ## 3. Delete all embedded links (ie. "https:....)
#
# For example see the first row in the tweets dataframe.
#
# *Note: this may require an google search :)*

# In[ ]:


# ## 4. Remove any leading or trailing white space in the tweet column

# In[ ]:


# ## 5. Rename the 'Party' column to 'Dem_Rep'
#
# No real reason here :) just wanted you to get practice doing this.

# In[ ]:


# ## 6. Concatenate the Party and Handle columns
#
# Silly yes... but good practice.
#
# pyspark.sql.functions.concat_ws(sep, *cols)[source] <br>
# Concatenates multiple input string columns together into a single string column, using the given separator.

# In[ ]:


# ## Challenge Question
#
# Let's image that we want to analyze the hashtags that are used in these tweets. Can you extract all the hashtags you see?

# In[ ]:


# # Let's create our own dataset to work with real dates
#
# This is a dataset of patient visits from a medical office. It contains the patients first and last names, date of birth, and the dates of their first 3 visits.

# In[3]:


from pyspark.sql.types import *

md_office = [
    ("Mohammed", "Alfasy", "1987-4-8", "2016-1-7", "2017-2-3", "2018-3-2"),
    ("Marcy", "Wellmaker", "1986-4-8", "2015-1-7", "2017-1-3", "2018-1-2"),
    ("Ginny", "Ginger", "1986-7-10", "2014-8-7", "2015-2-3", "2016-3-2"),
    ("Vijay", "Doberson", "1988-5-2", "2016-1-7", "2018-2-3", "2018-3-2"),
    ("Orhan", "Gelicek", "1987-5-11", "2016-5-7", "2017-1-3", "2018-9-2"),
    ("Sarah", "Jones", "1956-7-6", "2016-4-7", "2017-8-3", "2018-10-2"),
    ("John", "Johnson", "2017-10-12", "2018-1-2", "2018-10-3", "2018-3-2"),
]

df = spark.createDataFrame(
    md_office, ["first_name", "last_name", "dob", "visit1", "visit2", "visit3"]
)  # schema=final_struc

# Check to make sure it worked
df.show()
print(df.printSchema())


# Oh no! The dates are still stored as text... let's try converting them again and see if we have any issues this time.

# In[ ]:


# ## 7. Can you calculate a variable showing the length of time between patient visits?
#
# Compare visit1 to visit2 and visit2 to visit3 for all patients and see what the average length of time is between visits. Create an alias for it as well.

# In[ ]:


# ## 8. Can you calculate the age of each patient?

# In[ ]:


# ## 9. Can you extract the month from the first visit column and call it "Month"?

# In[ ]:


# ## 10. Challenges with working with date and timestamps
#
# Let's read in the supermarket sales dataframe attached to the lecture now and see some of the issues that can come up when working with date and timestamps values.

# In[ ]:


# ## About this dataset
#
# The growth of supermarkets in most populated cities are increasing and market competitions are also high. The dataset is one of the historical sales of supermarket company which has recorded in 3 different branches for 3 months data.
#
#  - Attribute information
#  - Invoice id: Computer generated sales slip invoice identification number
#  - Branch: Branch of supercenter (3 branches are available identified by A, B and C).
#  - City: Location of supercenters
#  - Customer type: Type of customers, recorded by Members for customers using member card and Normal for without member card.
#  - Gender: Gender type of customer
#  - Product line: General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health and beauty, Home and lifestyle, Sports and travel
#  - Unit price: Price of each product in USD
#  - Quantity: Number of products purchased by customer
#  - Tax: 5% tax fee for customer buying
#  - Total: Total price including tax
#  - Date: Date of purchase (Record available from January 2019 to March 2019)
#  - Time: Purchase time (10am to 9pm)
#  - Payment: Payment used by customer for purchase (3 methods are available – Cash, Credit card and Ewallet)
#  - COGS: Cost of goods sold
#  - Gross margin percentage: Gross margin percentage
#  - Gross income: Gross income
#  - Rating: Customer stratification rating on their overall shopping experience (On a scale of 1 to 10)
#
# **Source:** https://www.kaggle.com/aungpyaeap/supermarket-sales

# ### View dataframe and schema as usual

# In[ ]:


# In[ ]:


# ### Convert date field to date type
#
# Looks like we need to convert the date field into a date type. Let's go ahead and do that..

# In[ ]:


# ### How can we extract the month value from the date field?
#
# If you had trouble converting the date field in the previous question think about a more creative solution to extract the month from that field.

# In[ ]:


# ## 11.0 Working with Arrays
#
# Here is a dataframe of reviews from the movie the Dark Night.

# In[3]:


from pyspark.sql.functions import *

values = [
    (5, "Epic. This is the best movie I have EVER seen"),
    (4, "Pretty good, but I would have liked to seen better special effects"),
    (3, "So so. Casting could have been improved"),
    (
        5,
        "The most EPIC movie of the year! Casting was awesome. Special effects were so intense.",
    ),
    (4, "Solid but I would have liked to see more of the love story"),
    (5, "THE BOMB!!!!!!!"),
]
reviews = spark.createDataFrame(values, ["rating", "review_txt"])

reviews.show(6, False)


# ## 11.1 Let's see if we can create an array off of the review text column and then derive some meaningful results from it.
#
# **But first** we need to clean the rview_txt column to make sure we can get what we need from our analysis later on. So let's do the following:
#
# 1. Remove all punctuation
# 2. lower case everything
# 3. Remove white space (trim)
# 3. Then finally, split the string

# In[ ]:


# In[ ]:


# ## 11.2 Alright now let's see if we can find which reviews contain the word 'Epic'

# In[ ]:


# ### That's it! Great Job!

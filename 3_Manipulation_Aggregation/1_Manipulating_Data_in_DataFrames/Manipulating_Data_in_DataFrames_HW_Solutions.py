#!/usr/bin/env python
# coding: utf-8

# # Manipulating Data in DataFrames HW Solutions
#
#
# #### Let's get started applying what we learned in the lecure!
#
# I've provided several questions below to help test and expand you knowledge from the code along lecture. So let's see what you've got!
#
# First we will create our spark instance as we need to do at the start of every project.

# In[2]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Manip").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in our Republican vs. Democrats Tweet DataFrame

# In[22]:


path = "Datasets/"
tweets = spark.read.csv(path + "Rep_vs_Dem_tweets.csv", inferSchema=True, header=True)


# ## About this dataframe
#
# Extracted tweets from all of the representatives (latest 200 as of May 17th 2018)
#
# **Source:** https://www.kaggle.com/kapastor/democratvsrepublicantweets#ExtractedTweets.csv
#
# Let's check out the first view rows of the dataframe to get an idea of what we are working with.

# In[48]:


tweets.limit(4).toPandas()


# **Prevent Truncation of view**
#
# If the view you produced above truncated some of the longer tweets, see if you can prevent that so you can read the whole tweet.

# In[49]:


tweets.select("tweet").show(3, False)


# As we can see, this dataset contains three columns. The tweet content, Twitter handle that tweeted the tweet, and the party that that tweet belongs to. But it looks like the tweets could use some cleaning, esspecially if we are going to this for some kind of machine learning analysis. Let's see if we can make this an even richer dataset using the techniques we learned in the lecture!
#
# **Print Schema**
#
# First, check the schema to make sure the datatypes are accurate.

# In[4]:


print(tweets.printSchema())


# ## 1. Can you identify any tweet that mentions the handle @LatinoLeader using regexp_extract?
#
# It doesn't matter how you identify the row, any identifier will do. You can test your script on row 5 from this dataset. That row contains @LatinoLeader.

# In[76]:


from pyspark.sql.functions import *  # regexp_extract

latino = tweets.withColumn(
    "Latino_Mentions", regexp_extract(tweets.Tweet, "(.)(@LatinoLeader)(.)", 2)
)
latino.limit(6).toPandas()


# ## 2. Replace any value other than 'Democrate' or 'Republican' with 'Other' in the Party column.
#
# We can see from the output below, that there are several other values other than 'Democrate' or 'Republican' in the Part column. We are assuming that this is dirty data that needs to be cleaned up.

# In[9]:


# We haven't gotten to this yet so it's a bit of a teaser :)
from pyspark.sql.functions import *

counts = tweets.groupBy("Party").count()
counts.orderBy(desc("count")).show(6)


# In[47]:


from pyspark.sql.functions import when

clean = tweets.withColumn(
    "Party",
    when(tweets.Party == "Democrat", "Democrat")
    .when(tweets.Party == "Republican", "Republican")
    .otherwise("Other"),
)
counts = clean.groupBy("Party").count()
counts.orderBy(desc("count")).show(16)


# ## 3. Delete all embedded links (ie. "https:....)
#
# For example see the first row in the tweets dataframe.
#
# *Note: this may require an google search :)*

# In[73]:


print("OG Tweet")
tweets.select("tweet").show(1, False)


# In[39]:


# And here is the solution
print("Cleaned Tweet")
tweets.withColumn(
    "cleaned",
    regexp_replace(
        "Tweet",
        "(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
        "",
    ),
).select("cleaned").show(1, False)


# ## 4. Remove any leading or trailing white space in the tweet column

# In[4]:


from pyspark.sql.functions import *

tweets.select("Tweet").show(5, False)
tweets.select("Tweet", trim(tweets.Tweet)).show(5, False)


# ## 5. Rename the 'Party' column to 'Dem_Rep'
#
# No real reason here :) just wanted you to get practice doing this.

# In[75]:


renamed = tweets.withColumnRenamed("Party", "Dem_Rep")
renamed.limit(4).toPandas()


# ## 6. Concatenate the Party and Handle columns
#
# Silly yes... but good practice.
#
# pyspark.sql.functions.concat_ws(sep, *cols)[source] <br>
# Concatenates multiple input string columns together into a single string column, using the given separator.

# In[5]:


from pyspark.sql.functions import *

tweets.select(
    tweets.Party,
    tweets.Handle,
    concat_ws(" ", tweets.Party, tweets.Handle).alias("Concatenated"),
).show(5, False)


# ## Challenge Question
#
# Let's image that we want to analyze the hashtags that are used in these tweets. Can you extract all the hashtags you see?

# In[38]:


from pyspark.sql.functions import *

# Parenthesis are used to mark a subexpression within a larger expression
# The . matches any character other than a new line
# | means is like or
# \w+ means followed by any word
pattern = "(.|" ")(#)(\w+)"
# * is used to match the preceding character zero or more times.
# ? will match the preceding character zero or one times, but no more.
# $ is used to match the ending position in a string.
split_pattern = r".*?({pattern})".format(pattern=pattern)
end_pattern = r"(.*{pattern}).*?$".format(pattern=pattern)

# $1 here means to capture the first part of the regex result
# The , will separate each find with a comma in the a array we create
df2 = tweets.withColumn("a", regexp_replace("Tweet", split_pattern, "$1,")).where(
    col("Tweet").like("%#%")
)
df2.select("a").show(3, False)
# Remove all the other results that came up
df3 = df2.withColumn("a", regexp_replace("a", end_pattern, "$1"))
df3.select("a").show(3, False)
# Finally create an array from the result by splitting on the comma
df4 = df3.withColumn("a", split("a", r","))
df4.select("a").show(3, False)
df4.limit(3).toPandas()


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


# Ooops! The dates are still stored as text... let's try converting them again and see if we have any issues this time.

# In[4]:


# Covert the date columns into date types
df = (
    df.withColumn("dob", df["dob"].cast(DateType()))
    .withColumn("visit1", df["visit1"].cast(DateType()))
    .withColumn("visit2", df["visit2"].cast(DateType()))
    .withColumn("visit3", df["visit3"].cast(DateType()))
)

# Check to make sure it worked
df.show()
print(df.printSchema())


# # 7. Can you calculate a variable showing the length of time between patient visits?
#
# Compare visit1 to visit2 and visit2 to visit3 for all patients and see what the average length of time is between visits. Create an alias for it as well.

# In[8]:


from pyspark.sql.functions import *

diff1 = df.select(datediff(df.visit2, df.visit1).alias("diff"))
diff2 = df.select(datediff(df.visit3, df.visit2).alias("diff"))

# Append the two dataframes together
diff_combo = diff1.union(diff2)
diff_combo.show(5)


# # 8. Can you calculate the age of each patient?

# In[9]:


# We use the datediff function here as well
# And divide by 365 to get the age
# I also formated this value to get rid of all the decimal places
ages = df.select(format_number(datediff(df.visit1, df.dob) / 365, 1).alias("age"))
ages.show()


# ## 9. Can you extract the month from the first visit column and call it "Month"?

# In[11]:


month1 = df.select(month(df["visit1"]).alias("Month"))
month1.show(3)


# ## 10. Challenges with working with date and timestamps
#
# Let's read in our supermarket sales dataframe and see some of the issues that can come up when working with date and timestamps values.

# In[14]:


path = "Datasets/"
sales = spark.read.csv(path + "supermarket_sales.csv", inferSchema=True, header=True)


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

# In[15]:


sales.limit(6).toPandas()


# In[16]:


print(sales.printSchema())


# Looks like we need to convert the date field into a date type. Let's go ahead and do that..

# In[17]:


# I hope you liked this challenge!
# I could not get accurate results using the following two methods:
from pyspark.sql.types import *

print("This gives all null values")
df = sales.withColumn("Formatted Date", sales["Date"].cast(DateType()))
df = df.select("Date", "Formatted Date")
print(df.limit(6).toPandas())

print(" ")
print("This result gives the wrong results (notice that all months are january)")
sales.select(
    "Date",
    to_date(sales.Date, "mm/dd/yyyy").alias("Dateformatted"),
    month(to_date(sales.Date, "mm/dd/yyyy")).alias("Month"),
).show(5)


# So I will try to get creative with my solultions below


# ## How can we extract the month value from the date field?

# In[20]:


# We need to creative here
# First split the date field and get the month value
df = sales.select("Date", split(sales.Date, "/")[0].alias("Month"), "Total")

# Verify everything worked correctly
print("Verify")
df.show(5)
print(df.printSchema())


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

# In[4]:


# We can do 1-3 in one call here
df = reviews.withColumn(
    "cleaned_reviews",
    trim(lower(regexp_replace(col("review_txt"), "[^\sa-zA-Z0-9]", ""))),
)
df.show(1, False)


# In[5]:


# Then split on the spaces!
df = df.withColumn("review_txt_array", split(col("cleaned_reviews"), " "))
df.show(1, False)


# ## 11.2 Alright now let's see if we can find which reviews contain the word 'Epic'

# In[8]:


epic = df.withColumn("result", array_contains(col("review_txt_array"), "epic"))
epic.toPandas()


# ### That's it! Great Job!

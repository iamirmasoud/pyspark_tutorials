#!/usr/bin/env python
# coding: utf-8

# # Reading, Writing and Validating Data in PySpark HW
#
# Welcome to your first coding homework assignment in PySpark! I hope you enjoyed the lecture on Reading, Writing and Validating dataframes. Now it's time to put what you've learned into action!
#
# I've included several instructions below to help guide you through this homework assignment which I hope will get you feeling even comfortable reading, writing and validating dataframes. If you get stuck at any point, feel free to jump to the next lecture where I will guide you through my solutions to the HW assignment.
#
# Have fun!
#
# Let's dig right in!
#
#
# ## But first things first.....
# We need to always begin every Spark session by creating a Spark instance. Let's go ahead and use the method we learned in the lecture in the cell below. Also see if you can remember how to open the Spark UI (using a link that automatically guides you there).

# In[ ]:


# ## Next let's start by reading a basic csv dataset
#
# Download the pga_tour_historical dataset that is attached to this lecture and save it whatever folder you want, then read it in.
#
# **Data Source:** https://www.kaggle.com/bradklassen/pga-tour-20102018-data
#
# Rememer to try letting Spark infer the header and infer the Schema types!

# In[ ]:


# ## 1. View first 5 lines of dataframe
# First generate a view of the first 5 lines of the dataframe to get an idea of what is inside. We went over two ways of doing this... see if you can remember BOTH ways.

# In[ ]:


# ## 2. Print the schema details
#
# Now print the details of the dataframes schema that Spark infered to ensure that it was infered correctly. Sometimes it is not infered correctly, so we need to watch out!

# In[ ]:


# ## 3. Edit the schema during the read in
#
# We can see from the output above that Spark did not correctly infer that the "value" column was an integer value. Let's try specifying the schema this time to let spark know what the schema should be.
#
# Here is a link to see a list of PySpark data types in case you need it (also attached to the lecture):
# https://spark.apache.org/docs/latest/sql-ref-datatypes.html

# In[ ]:


# ## 4. Generate summary statistics for only one variable
#
# See if you can generate summary statistics for only the "Value" column using the .describe function
#
# (count, mean, stddev, min, max)

# In[ ]:


# ## 5. Generate summary statistics for TWO variables
# Now try to generate ONLY the count min and max for BOTH the "Value" and "Season" variable using the select. You can't use the .describe function for this one but see if you can remember which function you CAN use.

# In[ ]:


# ## 6. Write a parquet file
#
# Now try writing a parquet file (not partitioned) from the pga dataset. But first create a new dataframe containing ONLY the the "Season" and "Value" fields (using the "select command you used in the question above) and write a parquet file partitioned by "Season". This is a bit of a challenge aimed at getting you ready for material that will be covered later on in the course. Don't feel bad if you can't figure it out.
#
# *Note that if any of your variable names contain spaces, spark will produce an error message with this call. That is why we are selecting ONLY the "Season" and "Value" fields. Ideally we should renamed those columns but we haven't gotten to that yet in this course but we will soon!*

# In[ ]:


# ## 7. Write a partioned parquet file
#
# You will need to use the same limited dataframe that you created in the previous question to accomplish this task as well.

# In[ ]:


# ## 8. Read in a partitioned parquet file
#
# Now try reading in the partitioned parquet file you just created above.

# In[ ]:


# ## 9. Reading in a set of paritioned parquet files
#
# Now try only reading Seasons 2010, 2011 and 2012.

# In[ ]:


# ## 10. Create your own dataframe
#
# Try creating your own dataframe below using PySparks *.createDataFrame* function. See if you can make one that contains 4 variables and at least 3 rows.
#
# Let's see how creative you can get on the content of the dataframe :)

# In[ ]:


# ## We're done! Great job!

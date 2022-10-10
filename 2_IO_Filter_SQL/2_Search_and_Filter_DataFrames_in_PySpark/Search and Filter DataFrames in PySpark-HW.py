#!/usr/bin/env python
# coding: utf-8

# # Search and Filter DataFrames in PySpark HW

# Now it's time to put what you've learn into action with a homework assignment!
#
# In case you need it again, here is the link to the documentation for the full list available function in pyspark.sql.functions library:
# http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions
#
#
# ### First set up your Spark Session!
# Alright so first things first, let's start up our pyspark instance.

# In[ ]:


# ## Read in the DataFrame for this Notebook
#
# We will be continuing to use the fifa19.csv file for this notebook. Make sure that you are writting the correct path to the file.

# In[ ]:


# ## About this dataframe
#
# The **fifa19.csv** dataset includes a list of all the FIFA 2019 players and their attributes listed below:
#
#  - **General**: Age, Nationality, Overall, Potential, Club
#  - **Metrics:** Value, Wage
#  - **Player Descriptive:** Preferred Foot, International Reputation, Weak Foot, Skill Moves, Work Rate, Position, Jersey Number, Joined, Loaned From, Contract Valid Until, Height, Weight
#  - **Possition:** LS, ST, RS, LW, LF, CF, RF, RW, LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM, RWB, LB, LCB, CB, RCB, RB,
#  - **Other:** Crossing, Finishing, Heading, Accuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression, Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, and Release Clause.
#
# **Source:** https://www.kaggle.com/karangadiya/fifa19

# Use the .toPandas() method to view the first few lines of the dataset so we know what we are working with.

# In[ ]:


# Now print the schema of the dataset so we can see the data types of all the varaibles.

# In[ ]:


# ## Now let's get started!
#
# ### First things first..... import the pyspark sql functions library
#
# Since we know we will be using it a lot.

# In[ ]:


# ### 1. Select the Name and Position of each player in the dataframe

# In[ ]:


# ### 1.1 Display the same results from above sorted by the players names

# In[ ]:


# ### 2. Select only the players who belong to a club begining with FC

# In[ ]:


# In[ ]:


# ### 3. Who is the oldest player in the dataset and how old are they?
#
# Display only the name and age of the oldest player.

# In[ ]:


# ### 4. Select only the following players from the dataframe:
#
#  - L. Messi
#  - Cristiano Ronaldo

# In[ ]:


# ### 5. Can you select the first character from the Release Clause variable which indicates the currency used?

# In[ ]:


# ### 6. Can you select only the players who are over the age of 40?

# In[ ]:


# ### That's is for now... Great Job!

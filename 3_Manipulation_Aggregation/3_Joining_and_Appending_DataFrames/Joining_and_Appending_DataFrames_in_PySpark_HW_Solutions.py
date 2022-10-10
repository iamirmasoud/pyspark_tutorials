#!/usr/bin/env python
# coding: utf-8

# # Joining and Appending DataFrames in PySpark HW Solutions
#
# Now it's time to test your knowledge and further engrain the concepts we touched on in the lectures. Let's go ahead and get started.
#
#
#
#
# **As always let's start our Spark instance.**

# In[2]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("joins").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Read in the database
#
# Let cotinue working with our college courses dataframe to get some more insights and practice what we have learned!Let's read in the whole database using the loop function that we learned about in the lecture to automatically read in all the datasets from the uw-madision-courses folder (there are too many datasets to each one individually.

# In[3]:


import os

path = "Datasets/uw-madison-courses/"

df_list = []
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        filename_list = filename.split(".")  # separate path from .csv
        df_name = filename_list[0]
        df = spark.read.csv(path + filename, inferSchema=True, header=True)
        df.name = df_name
        df_list.append(df_name)
        exec(df_name + " = df")

# QA
print("Full list of dfs:")
print(df_list)


# Now check the contents of a few of the dataframes that were read in above.

# In[4]:


grade_distributions.limit(4).toPandas()


# ## Recap: About this database
#
# You will notice that there are several more tables in the uw-madision-courses folder than there are read in above. This so that you will have a chance to practice your own custom joins and learn about the relationships between a real database work. Sometimes we don't know how they are related and we need to figure it out! I'll save that for the HW :)
#
# Here is a look at some of the important variables we can use to join our tables:
#
#  - course_offerings: uuid, course_uuid, term_code, name
#  - instructors: id, name
#  - schedules: uuid
#  - sections: uuid, course_offering_uuid,room_uuid, schedule_uuid
#  - teachings: instructor_id, section_uuid
#  - courses: uuid
#  - grade_distributions: course_offering_uuid,section_number
#  - rooms: uuid, facility_code, room_code
#  - subjects: code
#  - subject_memberships: subject_code, course_offering_uuid
#
# **Source:** https://www.kaggle.com/Madgrades/uw-madison-courses
#
# So alright, let's use this information to discover some insights from this data!

# ## 1a. Can you assign the room numbers to each section of each course?
#
# Show only the rooms uuid, facility code, room number, term code and the name of the course from the course_offerings table.

# In[4]:


step1 = rooms.join(sections, rooms.uuid == sections.room_uuid, how="left").select(
    [rooms.uuid, rooms.facility_code, sections.course_offering_uuid, "number"]
)
step1.limit(4).toPandas()


# In[5]:


step2 = step1.join(
    course_offerings, step1.course_offering_uuid == course_offerings.uuid, how="left"
).select([rooms.uuid, rooms.facility_code, "number", "term_code", "name"])
step2.limit(4).toPandas()


# ## 1b. Now show same output as above but for only facility number 0469 (facility_code)

# In[6]:


step3 = step2.filter(step2.facility_code == "0469")
step3.limit(4).toPandas()


# ## 2. Count how many sections are offered for each subject for each facility
#
# *Note: this will involve a groupby*

# In[7]:


step1 = (
    subjects.join(
        subject_memberships,
        subjects.code == subject_memberships.subject_code,
        how="inner",
    )
    .select(["name", "course_offering_uuid"])
    .withColumnRenamed("name", "subject_name")
)
step1.limit(4).toPandas()


# In[5]:


step2 = step1.join(
    sections, step1.course_offering_uuid == sections.course_offering_uuid, how="left"
).select(["subject_name", "room_uuid"])
step2.limit(4).toPandas()


# In[20]:


# I added a filter to make this a little simpler
step3 = (
    step2.join(rooms, step2.room_uuid == rooms.uuid, how="left")
    .filter('facility_code IN("0140","0545","0469","0031")')
    .select(["subject_name", "facility_code", "room_code"])
)
step3.limit(4).toPandas()


# In[21]:


# Option 1: Group by facility code and do a count
step3.groupBy("facility_code", "subject_name").count().orderBy("facility_code").show(
    10, False
)  # False prevents truncation of column content


# In[24]:


# Option 2: Groupby subject name and pivot the facility code
# to see each facility side by side within each subject
step3.groupBy("subject_name").pivot("facility_code").count().show(10, False)


# ## 3. What are the hardest classes?
#
# Let's see if we can figure out which classes are the hardest by seeing how many students failed. Note that you will first need to aggregate the grades table by the course uuid to include all sections. Show the name of the course as well that you will need to get from the course_offering table.

# In[5]:


grade_distributions.limit(4).toPandas()


# In[9]:


course_offerings.limit(4).toPandas()


# In[12]:


step1 = grade_distributions.groupBy("course_offering_uuid").sum("f_count")
step1.limit(4).toPandas()


# In[22]:


step2 = (
    step1.join(
        course_offerings,
        step1.course_offering_uuid == course_offerings.uuid,
        how="left",
    )
    .select(["name", "sum(f_count)"])
    .orderBy("sum(f_count)")
)
step2.toPandas().tail(5)


# ## Challenge Question: Automating data entry errors
#
# We see in the dataframe below that there are several typos of various animal names. If this was a large database of several millions of records, correcting these errors would be way too labor intensive. How can we automate correcting these errors?
#
# *Hint: Leven...*

# In[8]:


values = [
    ("Monkey", 10),
    ("Monkay", 36),
    ("Mnky", 123),
    ("Elephant", 48),
    ("Elefant", 16),
    ("Ellafant", 1),
    ("Hippopotamus", 48),
    ("Hipopotamus", 16),
    ("Hippo", 1),
]
zoo = spark.createDataFrame(values, ["Animal", "age"])
zoo.show()


# In[14]:


# With the levenshtein distance!
from pyspark.sql.functions import *
from pyspark.sql.types import *

# First we create a dataframe with the 3 options we want to choose from
options = spark.createDataFrame(["Monkey", "Elephant", "Hippopotamus"], StringType())
options.show()


# In[16]:


# And then we join the two dataframes together with a condition >5
results = zoo.join(options, levenshtein(zoo["Animal"], options["value"]) < 5, "left")
results.show()


# So we can see here that all of our values were correctly identified except for "Hippo" which was just way too different from "Hippopotamus" to get correctly identified. So this solution won't work for EVERY case, but we can see here that it did a great job correcting simple gramatical errors.

# ### Great job!

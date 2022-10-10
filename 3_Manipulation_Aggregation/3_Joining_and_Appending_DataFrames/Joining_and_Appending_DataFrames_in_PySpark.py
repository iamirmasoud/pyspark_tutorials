#!/usr/bin/env python
# coding: utf-8

# # Joining and Appending DataFrames in PySpark
#
# In this lecture we will be reviewing the foundational concepts of joinings and appending dataframes as well as the necessary PySpark calls to accomplish these tasks. Many of you may already be familiar with the common join types we will review here, so I also want to spend some time going through how to really conceptualize the joining process from a foundational level so you can effectively apply these concepts to a real usecase which we will also do here. Understanding these concepts early on will help you in your day to day job, and more imporantly learning how to check your work and understand what you are doing help you to make less mistakes.
#
# So let's dig in!

# In[1]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("joins").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Generate play data
#
# First some play data to help us grasp some concepts. Let's create a database that has two tables.
#
# **Key Terms**
#  - **omnivore**: an animal which is able to consume both plants (like a herbivore) and meat (like a carnivore)
#  - **herbivore**: any animal that eats only vegetation (i.e. that eats no meat)
#  - **carnivore**: any animal that eats meat as the main part of its diet

# In[26]:


valuesP = [
    ("koala", 1, "yes"),
    ("caterpillar", 2, "yes"),
    ("deer", 3, "yes"),
    ("human", 4, "yes"),
]
eats_plants = spark.createDataFrame(valuesP, ["name", "id", "eats_plants"])

valuesM = [
    ("shark", 5, "yes"),
    ("lion", 6, "yes"),
    ("tiger", 7, "yes"),
    ("human", 4, "yes"),
]
eats_meat = spark.createDataFrame(valuesM, ["name", "id", "eats_meat"])

print("Plant eaters (herbivores)")
print(eats_plants.show())
print("Meat eaters (carnivores)")
print(eats_meat.show())


# ## Appends
#
# Appending "appends" two dataframes together that have the exact same variables. You can think of it like stacking two or more blocks ON TOP of each other. To demonstrate this, we will simply join the same dataframe to itself since we don't really have a good use case for this. But hopefully this will help you imagine what to do.
#
# A common usecase would be joining the same table of infomation from one year to another year (i.e. 2012 + 2013 + ...)

# In[27]:


# So first replicate table and call it new_df
new_df = eats_plants
# Then append using the union function
# this naming convention can be tricky to grasp for SQL enthusiasts
# Where union just mean join
df_concat = eats_plants.union(new_df)
# We will test to see if this worked by getting before and after row counts
print(("eats_plants df Counts:", eats_plants.count(), len(eats_plants.columns)))
print(("df_concat Counts:", df_concat.count(), len(df_concat.columns)))
print(eats_plants.show(5))
print(df_concat.show(5))


# ## Inner Joins!
#
# Inner joins get us ONLY the values that appear in BOTH tables we are joining.

# In[29]:


inner_join = eats_plants.join(eats_meat, ["name", "id"], "inner")
print("Inner Join Example")
print(inner_join.show())
# So this is the only name that appears in BOTH dataframes


# ## Left Joins
#
# Left joins get us the values that appear in the left table and nothing additional from the right table except for its columns. A quick quality check we could do would be to make sure that the human column has the value "yes" for both eats_plants and eats_meat columns.

# In[30]:


left_join = eats_plants.join(
    eats_meat, ["name", "id"], how="left"
)  # Could also use 'left_outer'
print("Left Join Example")
print(left_join.show())


# ## Conditional Joins
#
# Conditional joins have some additional logic that was not encompassed in the underlying join. For example, if we wanted to get all the values that appear in the left, **except** for those values that appear in BOTH tables, we could do this. Notice how human is left out now.

# In[31]:


conditional_join = eats_plants.join(eats_meat, ["name", "id"], how="left").filter(
    eats_meat.name.isNull()
)
print("Conditional Left Join")
print(conditional_join.show())


# ## Right Join
#
# A right join gets you the values that appear in the right table but not in the left. It also brings it's columns over of course.

# In[7]:


right_join = eats_plants.join(
    eats_meat, ["name", "id"], how="right"
)  # Could also use 'right_outer'
print("Right Join")
print(right_join.show())


# ## Full Outer Joins
#
# Full outer joins will get all values from both tables, but notice that if there is a column that is common in both tables (ie. id and name in this case) that the join will take the value of the left table (see human id is p4 and not m4).

# In[10]:


full_outer_join = eats_plants.join(
    eats_meat, ["name", "id"], how="full"
)  # Could also use 'full_outer'
print("Full Outer Join")
print(full_outer_join.show())


# ## Alright now let's try with REAL data
#
# Thinking about how to join your data in real life will not be as easy as the above. You need to consider multiple aspects as you join tables in real life and ALWAYS conduct sanity checks to make sure you did it correctly. Let's look at an example below with real data.
#
# #### First, let's read in the datasets we will be working with
#
# Here is a neat function that will read in all the csv files from a directory (folder) in one shot and returns a separate dataframe for each dataset in the directory using the same naming convention. This is super useful if you have a large set of files and don't feel like writing a separate line for each dataset in the directory.

# In[11]:


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


# ## About this database
#
# You will notice that there are several tables in the uw-madision-courses folder that were read in above. This database will let you get a chance to practice your own custom joins and learn how the relationships between a real database work. Sometimes we don't know how they are related and we need to figure it out! I'll save that for the HW since we will be using the same database :) So I just wanted to introduce the database to you quickly here first.
#
# For this lecture, we will focus on the 4 datasets below and save the rest for the HW. Here is a look at some of the important variables we will be using to join our tables:
#
#  - **course_offerings:** uuid, course_uuid, term_code, name
#  - **instructors:** id, name
#  - **sections:** uuid, course_offering_uuid,room_uuid, schedule_uuid
#  - **teachings:** instructor_id, section_uuid
#
#  **Source:** https://www.kaggle.com/Madgrades/uw-madison-courses
#
# Let's pretend that I am a student interested in seeing what courses are available. I suppose I would start by look at the course offerings table.

# In[12]:


# View the data
course_offerings.limit(4).toPandas()


# This course offers table is great, but I also want to know who teaches each course because I want to check the reviews of the instructor before I take the course. Let's see if we can join this table with the instructors table that contains the name of the instructor.

# In[16]:


instructors.show(4, False)


# Hmmm, so this table only contains 2 columns (id and name) and doesn't have the uuid or course uuid to join on. So we will need to see how we can accomplish the join we need. It looks like from the tables we have, we would need to take the following steps to get the variables we need.
#
#  - **course_offerings (CO):** uuid, course_uuid, term_code, name
#  - **instructors (I):** id, name
#  - **sections (S):** uuid, course_offering_uuid,room_uuid, schedule_uuid
#  - **teachings (T):** instructor_id, section_uuid
#
#  I.id --> T.instructor_id
#                 \/
#           T.section_uuid --> S.uuid
#                               \/
#                              S.course_offering_uuid --> CO.uuid

# In[19]:


teachings.show(3)


# In[13]:


# Let's try to see all course offerings and who teaches it
# Notice here that the variable we want to join on is different in the two datasets.
# PySpark makes it easy to account for that
step1 = teachings.join(
    instructors, teachings.instructor_id == instructors.id, how="left"
).select(["instructor_id", "name", "section_uuid"])
step1.limit(4).toPandas()


# In[14]:


step2 = step1.join(sections, step1.section_uuid == sections.uuid, how="left").select(
    ["name", "course_offering_uuid"]
)
step2.limit(4).toPandas()


# In[15]:


step3 = (
    step2.withColumnRenamed("name", "instructor")
    .join(
        course_offerings,
        step2.course_offering_uuid == course_offerings.uuid,
        how="inner",
    )
    .select(["instructor", "name", "course_offering_uuid"])
)
step3.limit(4).toPandas()


# And that's it! Sometimes it's helpful to think through joins step by step like this. I hope that helped get the concept down.
#
# ## One final really cool way to join datasets: The Levenshtien distance!
#
# Which basically counts the number of edits you would need to make to make too strings equal to eachother. I'll let you figure the joining part in the HW!

# In[ ]:


# Compute the levenshtein distance beween two strings
# pyspark.sql.functions.levenshtein(left, right)

from pyspark.sql.functions import levenshtein

df0 = spark.createDataFrame(
    [("Aple", "Apple", "Microsoft", "IBM")], ["Input", "Option1", "Option2", "Option3"]
)
print("Correct this company name: Aple")
df0.select(levenshtein("Input", "Option1").alias("Apple")).show()
df0.select(levenshtein("Input", "Option2").alias("Microsoft")).show()
df0.select(levenshtein("Input", "Option3").alias("IBM")).show()


# ### Great job!

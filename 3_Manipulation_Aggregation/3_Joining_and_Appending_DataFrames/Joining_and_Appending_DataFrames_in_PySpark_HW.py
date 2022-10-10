#!/usr/bin/env python
# coding: utf-8

# # Joining and Appending DataFrames in PySpark HW
#
# Now it's time to test your knowledge and further engrain the concepts we touched on in the lectures. Let's go ahead and get started.
#
#
#
#
# **As always let's start our Spark instance.**

# In[ ]:


# ## Read in the database
#
# Let cotinue working with our college courses dataframe to get some more insights and practice what we have learned!Let's read in the whole database using the loop function that we learned about in the lecture to automatically read in all the datasets from the uw-madision-courses folder (there are too many datasets to each one individually.

# In[ ]:


# Now check the contents of a few of the dataframses that were read in above.

# In[ ]:


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
#  **Source:** https://www.kaggle.com/Madgrades/uw-madison-courses
#
# So alright, let's use this information to discover some insights from this data!

# ## 1a. Can you assign the room numbers to each section of each course?
#
# Show only the rooms uuid, facility code, room number, term code and the name of the course from the course_offerings table.

# In[ ]:


# In[ ]:


# ## 1b. Now show same output as above but for only facility number 0469 (facility_code)

# In[ ]:


# ## 2. Count how many sections are offered for each subject for each facility
#
# *Note: this will involve a groupby*

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# ## 3. What are the hardest classes?
#
# Let's see if we can figure out which classes are the hardest by seeing how many students failed. Note that you will first need to aggregate the grades table by the course uuid to include all sections. Show the name of the course as well that you will need to get from the course_offering table.

# In[ ]:


# In[ ]:


# ## Challenge Question: Automating data entry errors
#
# We see in the dataframe below that there are several typos of various animal names. If this was a large database of several millions of records, correcting these errors would be way too labor intensive. How can we automate correcting these errors?
#
# *Hint: Leven...*

# In[ ]:


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


# In[ ]:


# In[ ]:


# In[ ]:


# ### Great job!

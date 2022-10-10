#!/usr/bin/env python
# coding: utf-8

# # Regression in PySpark's MLlib Project
#
# Now it's time to put what you've learned to into action with a REAL project!
#
# You have been hired as a consultant to a cement production company who wants to be able to improve their customer experience around a number of areas like being able to provide recommendations to cusomters on optimal amounts of certian ingredients in the cement making process and perhaps even create an application where users can input their own values and received a predicted cement strength!
#
# I have provided a list of question below to help guide you through this project but feel free to deviate and make this project your own! But first, a bit about this dataset.
#
# ### About this dataset
# This dataset contains 1030 instances of concrete samples, containing 9 attributes (8 continuous and 1 discreate), and 1 continuous quantitative output variable. There are no missing attribute values.
#
# I've also provided the variable name, variable type, the measurement unit and a brief description of each variable in the dataset. The concrete compressive strength is the outcome variable for our analysis. The order of this listing corresponds to the order of numerals along the rows of the database.
#
# Name -- Data Type -- Measurement -- Description
#
# - Cement -- quantitative -- kg in a m3 mixture -- Input Variable
# - Blast Furnace Slag -- quantitative -- kg in a m3 mixture -- Input Variable
# - Fly Ash -- quantitative -- kg in a m3 mixture -- Input Variable
# - Water -- quantitative -- kg in a m3 mixture -- Input Variable
# - Superplasticizer -- quantitative -- kg in a m3 mixture -- Input Variable
# - Coarse Aggregate -- quantitative -- kg in a m3 mixture -- Input Variable
# - Fine Aggregate -- quantitative -- kg in a m3 mixture -- Input Variable
# - Age -- quantitative -- Day (1~365) -- Input Variable
# - Concrete compressive strength -- quantitative -- MPa -- Output Variable
#
# **Source:** https://www.kaggle.com/maajdl/yeh-concret-data
#
# **Dataset Name:** Concrete_Data.csv

# ## 1. Which features are the strongest predictors of cement strength?
#
# Build your own ML model to figure this one out! This would be good information to give to our client so the sales reps can focus their efforts on certian ingredients to provide recommendations on. For example, if our clients had a customer that was struggling with their cement breaking, we could trouble shoot with them by starting with the factors that we know are important.

# ## 2. For the following given inputs, what would be the estimated cement strength?
#
# - Cement: 540
# - Blast Furnace Slag: 0
# - Fly Ash: 0
# - Water: 162
# - Superplasticizer: 2.5
# - Coarse Aggregate: 1040
# - Fine Aggregate: 676
# - Age: 28
#
# The correct answer is 79.99. Let's how close your prediction is!

# ## 3. Now see if you can ask users to input their own value for Age and return a predicted value for the cement stength.
#
# We did not cover this is in the lecture so you'll have to put your thinking cap on. Accepting user input in PySpark works just like it does in traditional Python.
# <br>
#
# val = input("Enter your value: ")

# ## 4. Make recommendations of optimal values for cement ingredients (our features)
#
# See if you can find the optimal amount of cement to recommend holding the rest of the values from the previous question constant, assuming that the higher the cement strength value the better.

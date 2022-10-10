#!/usr/bin/env python
# coding: utf-8

# # Frequent Pattern Mining in PySpark's MLlib Project
#
# Let's see if you can use the concepts we learned about in the lecture to try out frequent pattern mining techniques on a new dataset!
#
#
# ## Recap:
#
# Spark MLlib provides two algorithms related to frequent pattern mining (FPM):
#
# 1. FP-growth (Frequent Pattern Growth)
# 2. PrefixSpan
#
# The distinction is that FP-growth does not use order information in the itemsets, if any, while PrefixSpan is designed for sequential pattern mining where the itemsets are ordered.
#
# ## Data
#
# You own a mall and through membership cards, you have some basic data about your customers like Customer ID, age, gender, annual income and spending score. Spending Score is something you assign to the customer based on your defined parameters like customer behavior and purchasing data.
#
# **Dataset:** Mall_Customers.csv <br>
# **Data Source:**  https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
#
# ## Problem statement
#
# As the mall owner, you want to understand the customers who can be easily grouped together so that a strategy can be provided to the marketing team to plan accordingly.
#
# *Note:* <br>
# You may need to transform the data in a way that will be meaningful for your market analysis. Think about how you might group the customers in the this data.
#
# You will also notice that I did not provide any leading questions in this notebook as I usually do. This to provide a bit of a challenge for you as this is the last concept we will be covering before the final project! I hope you enjoy the challenge :)

# In[ ]:

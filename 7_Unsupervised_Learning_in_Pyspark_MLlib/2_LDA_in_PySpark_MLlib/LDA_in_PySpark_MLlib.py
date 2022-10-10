#!/usr/bin/env python
# coding: utf-8

# # LDA in PySparks MLlib
#
# **Linear Discriminant Analysis**, otherwise known as *Topic Modeling*, is an unsupervised machine learning method for discovering topics by reducing dimension space. This method is very popular with Natural Language Processing (NLP) where the intent of the analysis is to discover themes or tags in a dataset that contains natural written language like tweets, news paper articles or even recipes!
#
# Today we will be analyzing a collection of christmas cooking recipes scraped from BBC Good Food. Let's see if we can try to discover some additional themes amoungst these recipes imagining that we want to create our own website that provides a more intelligent tagging system to recipes that are pulled from multiple data sources.
#
# **Approach** <br>
# There are several ways to tackle Topic Modeling out there today so I thought I would spend today introducing you to one of the most popular and traditional approaches just get you comfortable with the concept. The coding leading up to where we pass our data into the LDA algorithm will look much like what we covered in the NLP series where we first cleaned our data, then tokenize the data, remove stopwords, and then create a count vector. Then the next step of our analysis will start to look more like the approach we took in the k-means lecture where we will need to find the "optimal k" which in this case we can think of more lore like topics instead of clusters. Then the fun part starts where we can see that kinds of topics our model came up with!
#
# So let's dig in and see what it finds!

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("LDA").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
appid = spark._jsc.sc().applicationId()
print("You are working with", cores, "core(s) on appid: ", appid)
spark


# In[2]:


# For pipeline development in case you need it (but we won't use it here)
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA

# from pyspark import SparkContext
# from pyspark.sql import SQLContext
# from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import *
from pyspark.sql.functions import *

# Read in dependencies
# import string
from pyspark.sql.types import *

# ## Dataset
#
# This JSON lines file contains 1600 christmas cooking recipes scraped from BBC Good Food.
#
# ### Content
#
#  - Recipe Title
#  - Recipe Description (we are going to focus on this variable for our analysis)
#  - Recipe Author
#  - Ingredients list
#  - Step by step method
#
# **Source:** https://www.yanlong.app/gjbroughton/christmas-recipes
#
#
#
# **Read in the data**

# In[3]:


path = "Datasets/"
df = spark.read.json(path + "recipes.json")


# **View data as always**

# In[4]:


df.limit(5).toPandas()


# **And of course the schema**

# In[5]:


df.printSchema()


# ## Investigate missing values

# In[6]:


from pyspark.sql.functions import *


def null_value_calc(df):
    null_columns_counts = []
    numRows = df.count()
    for k in df.columns:
        nullRows = df.where(col(k).isNull()).count()
        if nullRows > 0:
            temp = k, nullRows, (nullRows / numRows) * 100
            null_columns_counts.append(temp)
    return null_columns_counts


null_columns_calc_list = null_value_calc(df)
spark.createDataFrame(
    null_columns_calc_list, ["Column_Name", "Null_Values_Count", "Null_Value_Percent"]
).show()


# ## Drop missing values
#
# Since we are focusing on the description variable for our analysis today, I'm just going to drop the rows that have missing values in that column.

# In[7]:


# drop missing values for the sake of the example
df = df.na.drop(subset=["Description"])
df.count()


# ## Prep the data (the NLP part)
#
# This will include the following steps:
#
#  - Clean text
#  - Tokenize
#  - Remove Stopwords
#  - Create Count Vector of your words (features)
#

# In[8]:


# Let's see a few full rows for the description column
df.select("Description").show(2, False)


# Looks like pretty standard text with punctuation, camel casing and hyphenated words. We can use a pretty simple approach for this using two of the regex replace methods we saw in the NLP lecture and the lower method which lower cases all the text.

# In[9]:


############## Clean text ##############

# Removing anything that is not a letter
df_clean = df.withColumn(
    "Description", lower(regexp_replace(col("Description"), "[^A-Za-z ]+", " "))
)
# Remove multiple spaces (because we replaced punctuation with a space)
df_clean = df_clean.withColumn(
    "Description", regexp_replace(col("Description"), " +", " ")
)
df_clean.select("Description").show(2, False)


# **Tokenize, remove stopwords and create a count vector**
#
# We will use a pipeline for this.

# In[10]:


# Tokenize
regex_tokenizer = RegexTokenizer(
    inputCol="Description", outputCol="words", pattern="\\W"
)
raw_words = regex_tokenizer.transform(df_clean)

# Remove Stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
words_df = remover.transform(raw_words)

# Zero Index Label Column
cv = CountVectorizer(inputCol="filtered", outputCol="features")
cvmodel = cv.fit(words_df)
df_vect = cvmodel.transform(words_df)


# In[11]:


# View the new dataframe
df_vect.limit(2).toPandas()


# ## Determine Optimal K for LDA
#
# This portion is going to be more like our k-means analysis where we plot our results and look for the elbow except this time we will be using the following metrics (the only ones PySpark provides):
#
# **Log likelihood:** Calculates a lower bound on the log likelihood of the entire corpus. We want higher numbers here. See Equation (16) in the Online LDA paper (Hoffman et al., 2010).
#
# **Log Perplexity:** Calculate an upper bound on perplexity. See Equation (16) in the Online LDA paper (Hoffman et al., 2010). We want lower numbers here, however keep in mind that this evaluation metric is debatable as many have shown that, surprisingly, perplexity and human judgment are often not correlated, and even sometimes slightly anti-correlated. Here is one interesting paper here on the subject: http://qpleple.com/perplexity-to-evaluate-topic-models/

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

kmax = 30
ll = np.zeros(kmax)
lp = np.zeros(kmax)
for k in range(2, kmax):
    lda = LDA(k=k, maxIter=10)
    model = lda.fit(df_vect)
    ll[k] = model.logLikelihood(df_vect)
    lp[k] = model.logPerplexity(df_vect)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, kmax), ll[2:kmax])
ax.set_xlabel("k")
ax.set_ylabel("ll")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(2, kmax), lp[2:kmax])
ax.set_xlabel("k")
ax.set_ylabel("lp")


# ### Conclusions from graph output obove
# Looks like in both cases (ll and lp) our model gets worse as k increases. So let's go with a low number of k. I'm going to go with 4 for the purpose of demonstration but we might want to play around with this a bit in a real analysis.
#
#
# ## Train your Final LDA Model using preferred K

# In[13]:


# Trains a LDA model.
lda = LDA(k=4, maxIter=10)
model = lda.fit(df_vect)


# Here is where things get fun!

# In[17]:


print("Recap of ll and lp:")
ll = model.logLikelihood(df_vect)
lp = model.logPerplexity(df_vect)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))
print("Vocab Size: ", model.vocabSize())

# Describe topics.
print("The topics described by their top-weighted terms:")
topics = model.describeTopics(maxTermsPerTopic=4)
# topics.show(truncate=False)
topics = topics.collect()
vocablist = cvmodel.vocabulary
# Enumerate adds a counter to topics and returns it in a form of enumerate object.
for x, topic in enumerate(topics):
    print(" ")
    print("TOPIC: " + str(x))
    # This is like a temp holder
    topic = topics
    # Then we extract the words from the topics
    words = topic[x][1]
    # Then print the words by topics
    for n in range(len(words)):
        print(vocablist[words[n]])  # + ' ' + str(weights[n])


# Cool! Looks like there are some relativley interesting topics but not super clear. From here we would need to create our own naming convention for each topic, like for example, topic 2 might be "Traditional Chocolate Cake Recipes". Or you could add tags to the recipes. Now let's see if we can recommend a topic for each row in the dataframe.
#
#
# ## Recommend a Topic for each row

# In[15]:


# Make predictions
transformed = model.transform(df_vect)
transformed.toPandas()

# Convert topicdistribution col from vector to array
# We need to create a udf for this one
to_array = udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))
recommendations = transformed.withColumn("array", to_array("topicDistribution"))

# Find the best topic value that we will call "max"
max_vals = recommendations.withColumn("max", array_max("array"))

# Find the index of the max value found above which translates to our topic!
argmaxUdf = udf(lambda x, y: [i for i, e in enumerate(x) if e == y])
results = max_vals.withColumn("topic", argmaxUdf(max_vals.array, max_vals.max))
results.printSchema()
results.limit(4).toPandas()


# ## Next Steps
#
# Awesome! Now that we have our data in a manageable form, we could use this to autocategorize recipes in a new recipe site we create into meaningful tabs like "chocolate" or "baking". We could also just auto-tag any recipes that are uploaded by users too which would also users to search our database more easily. How cool would that be?! It might also be cool look at the most common topics by Author or common ingredient.

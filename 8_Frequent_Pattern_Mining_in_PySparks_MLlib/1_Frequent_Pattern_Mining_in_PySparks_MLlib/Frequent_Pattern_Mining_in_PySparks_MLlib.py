#!/usr/bin/env python
# coding: utf-8

# # Frequent Pattern Mining in PySpark's MLlib
#
# As we discussed in the concept review lecture, PySpark offers two algorithms for frequency pattern mining (FPM):
#
# - FP-growth
# - PrefixSpan
#
# The distinction is that FP-growth does not use order information in the itemsets, if any, while PrefixSpan is designed for sequential pattern mining where the itemsets are ordered.

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("FPM").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark
# Click the hyperlinked "Spark UI" link to view details about your Spark session


# ## Import data
#
# ### About dataframe
#
# This data was collected (2016-2018) through an interactive on-line personality test.
# The personality test was constructed with the "Big-Five Factor Markers" from the IPIP. https://ipip.ori.org/newBigFive5broadKey.htm
# Participants were informed that their responses would be recorded and used for research at the beginning of the test, and asked to confirm their consent at the end of the test.
#
# The following items were presented on one page and each was rated on a five point scale using radio buttons. The order on page was was EXT1, AGR1, CSN1, EST1, OPN1, EXT2, etc.
# The scale was labeled 1=Disagree, 3=Neutral, 5=Agree
#
#  - EXT1	I am the life of the party.
#  - EXT2	I don't talk a lot.
#  - EXT3	I feel comfortable around people.
#  - EXT4	I keep in the background.
#  - EXT5	I start conversations.
#  - EXT6	I have little to say.
#  - EXT7	I talk to a lot of different people at parties.
#  - EXT8	I don't like to draw attention to myself.
#  - EXT9	I don't mind being the center of attention.
#  - EXT10	I am quiet around strangers.
#  - EST1	I get stressed out easily.
#  - EST2	I am relaxed most of the time.
#  - EST3	I worry about things.
#  - EST4	I seldom feel blue.
#  - EST5	I am easily disturbed.
#  - EST6	I get upset easily.
#  - EST7	I change my mood a lot.
#  - EST8	I have frequent mood swings.
#  - EST9	I get irritated easily.
#  - EST10	I often feel blue.
#  - AGR1	I feel little concern for others.
#  - AGR2	I am interested in people.
#  - AGR3	I insult people.
#  - AGR4	I sympathize with others' feelings.
#  - AGR5	I am not interested in other people's problems.
#  - AGR6	I have a soft heart.
#  - AGR7	I am not really interested in others.
#  - AGR8	I take time out for others.
#  - AGR9	I feel others' emotions.
#  - AGR10	I make people feel at ease.
#  - CSN1	I am always prepared.
#  - CSN2	I leave my belongings around.
#  - CSN3	I pay attention to details.
#  - CSN4	I make a mess of things.
#  - CSN5	I get chores done right away.
#  - CSN6	I often forget to put things back in their proper place.
#  - CSN7	I like order.
#  - CSN8	I shirk my duties.
#  - CSN9	I follow a schedule.
#  - CSN10	I am exacting in my work.
#  - OPN1	I have a rich vocabulary.
#  - OPN2	I have difficulty understanding abstract ideas.
#  - OPN3	I have a vivid imagination.
#  - OPN4	I am not interested in abstract ideas.
#  - OPN5	I have excellent ideas.
#  - OPN6	I do not have a good imagination.
#  - OPN7	I am quick to understand things.
#  - OPN8	I use difficult words.
#  - OPN9	I spend time reflecting on things.
#  - OPN10	I am full of ideas.
#
# The time spent on each question is also recorded in milliseconds. These are the variables ending in _E. This was calculated by taking the time when the button for the question was clicked minus the time of the most recent other button click.
#
# dateload    The timestamp when the survey was started.
# screenw     The width the of user's screen in pixels
# screenh     The height of the user's screen in pixels
# introelapse The time in seconds spent on the landing / intro page
# testelapse  The time in seconds spent on the page with the survey questions
# endelapse   The time in seconds spent on the finalization page (where the user was asked to indicate if they has answered accurately and their answers could be stored and used for research. Again: this dataset only includes users who answered "Yes" to this question, users were free to answer no and could still view their results either way)
# IPC         The number of records from the user's IP address in the dataset. For max cleanliness, only use records where this value is 1. High values can be because of shared networks (e.g. entire universities) or multiple submissions
# country     The country, determined by technical information (NOT ASKED AS A QUESTION)
# lat_appx_lots_of_err    approximate latitude of user. determined by technical information, THIS IS NOT VERY ACCURATE. Read the article "How an internet mapping glitch turned a random Kansas farm into a digital hell" https://splinternews.com/how-an-internet-mapping-glitch-turned-a-random-kansas-f-1793856052 to learn about the perils of relying on this information
# long_appx_lots_of_err   approximate longitude of user
#
#
# **Source:** https://www.kaggle.com/tunguz/big-five-personality-test#data-final.csv

# In[4]:


path = "Datasets/Big Five Personality Test/"
df = spark.read.option("delimiter", "\t").csv(
    path + "data-final.csv", inferSchema=True, header=True
)


# In[5]:


df.limit(4).toPandas()


# In[113]:


df.printSchema()


# In[114]:


# How many rows do we have in our dataframe?
df.count()


# ## Find frequent patterns (unordered)
#
# Using the FPGrowth Model
#
# In order to fit an FPGrowth model, our data needs to not have any duplicative entries within each row (ex. 1,2,3,1). Therefore we need to recode our values to some way that will have unique values in each row. Let's go ahead do that in a bit of a creative way. I want to know what each persons min, median and max values were for each question in the Ext group.

# In[167]:


from pyspark.sql.functions import *

p_types = df.withColumn(
    "vert",
    expr(
        "CASE WHEN EXT1 in('4','5') or EXT5 in('4','5') or EXT7 in('4','5') or EXT9 in('4','5') THEN 'extrovert' WHEN EXT1 in('1','2') or EXT5 in('1','2') or EXT7 in('1','2') or EXT9 in('1','2') THEN 'introvert' ELSE 'neutrovert' END AS vert"
    ),
)
p_types = p_types.withColumn(
    "mood",
    expr(
        "CASE WHEN EST2 in('4','5') THEN 'chill' WHEN EST2 in('1','2') THEN 'highstrung' ELSE 'neutral' END AS mood"
    ),
)

p_types = p_types.select(array("mood", "vert").alias("items"))
p_types.limit(4).toPandas()


# **Fit the model**

# In[172]:


from pyspark.ml.fpm import FPGrowth

fpGrowth = FPGrowth(itemsCol="items", minSupport=0.3, minConfidence=0.1)
model = fpGrowth.fit(p_types)


# ## Determine item popularity
#
# See what combos were most popular

# In[173]:


itempopularity = model.freqItemsets
itempopularity.createOrReplaceTempView("itempopularity")
# Then Query the temp view
print("Top 20")
spark.sql("SELECT * FROM itempopularity ORDER BY freq desc").limit(200).toPandas()


# ## Review Association Rules
#
# In addition to freqItemSets, the FP-growth model also generates **associationRules**. For example, if a shopper purchases peanut butter, what is the probability (or confidence) that they will also purchase jelly.  For more information, a good reference is Susan Li’s *A Gentle Introduction on Market Basket Analysis — Association Rules*
#
# A good way to think about association rules is that model determines that if you purchased something (i.e. the antecedent), then you will purchase this other thing (i.e. the consequent) with the following confidence.
#
# **Source:** https://databricks.com/blog/2018/09/18/simplify-market-basket-analysis-using-fp-growth-on-databricks.html

# In[174]:


# Display generated association rules.
assoc = model.associationRules
assoc.createOrReplaceTempView("assoc")
# Then Query the temp view
print("Top 20")
spark.sql("SELECT * FROM assoc ORDER BY confidence desc").limit(20).toPandas()


# ## Predict the consequent on your dataframe
#
# You can also transform your original dataframe, or any for that matter, to try to predict the consequent. This is super useful if you want to try to recommend an additional product to a shopper based on what they currently have in their cart.

# In[176]:


# transform examines the input items against all the association rules and summarize the
# consequents as prediction
predict = model.transform(p_types)
predict.limit(15).toPandas()


# ### Take away
#
# Great start! We would obviously want to expand our analysis to encompass all of the variables in the dataset but I hope you understand the logic here. We could use this analysis to do any of the following:
#
#  - These results could be used group together teams that need to be well rounded like half extroverts and half introverts, or the converse if you want to group people by their personality like say sales people need to extroverts.
#  - You could create an application that would ask respondents to answers these same questions and then provide an output that would be your daignosis of their personality type and perhaps even provide advice based on that finding. For example, types of jobs they would be well suited for or relationship advice.

# ## Now try to find patterns where the order DOES matter
#
# *Using the prefixspan model*
#
# If you examine the EXT statements closely, you will realize that every odd numbered variable is an extrovert statement (ie. EXT1, EXT3, EXT5, etc.) and every even numbered variable is an introvert statement (i.e. EXT2, EXT4,EXT6, etc.). Furthermore, each of the EXT statements can be paired with it's alternate (ie. EXT1 and EXT2), which are converse to each other. So in theory, a person who answers 5 to EXT1, would chose 1 for EXT2 or something similar.
#
#  - EXT1	I am the life of the party.
#  - EXT2	I don't talk a lot.
#  - EXT3	I feel comfortable around people.
#  - EXT4	I keep in the background.
#  - EXT5	I start conversations.
#  - EXT6	I have little to say.
#  - EXT7	I talk to a lot of different people at parties.
#  - EXT8	I don't like to draw attention to myself.
#  - EXT9	I don't mind being the center of attention.
#  - EXT10	I am quiet around strangers.
#
#
# So a person who answers [5,1] to all the paired EXT questions would be an extreme extrovert and someone who answers [1,5] to all the paired EXT questions, would be an extreme introvert, and everyone else would be some other variation of this.

# In[182]:


df_array = df.select(
    array(
        array("EXT1", "EXT2"),
        array("EXT3", "EXT4"),
        array("EXT5", "EXT6"),
        array("EXT7", "EXT8"),
        array("EXT9", "EXT10"),
    ).alias("sequence")
)
df_array.show(truncate=False)


# #### Apply the PrefixSpan algorithm
#
# and find the frequent patterns

# In[201]:


from pyspark.ml.fpm import PrefixSpan

prefixSpan = PrefixSpan(minSupport=0.3, maxPatternLength=10)

# Find frequent sequential patterns.
sequence = prefixSpan.findFrequentSequentialPatterns(df_array)
sequence.show(10)


# From the above output, we can see that it's finding patterns of even the single occurance arrays which is not super helpful. We can filter on the size of entire sequence like this if we want....

# In[196]:


sequence.where(size(col("sequence")) > 1).show()


# But that's not super helpful either. What we need to do is find the size (formerly known as length in python), of each array within the sequence array and filter on that.
#
# like this...

# In[210]:


from pyspark.sql.functions import expr, round

# get the size of each array within the arrays
filtered = sequence.withColumn("size", expr("transform(sequence, x -> size(x))"))
# sequence.withColumn("length",size(col("sequence")))

# Let's also add a column that tells us the percentage of the sequences
row_cnt = df_array.count()
filtered = filtered.withColumn("percentage", round((col("freq") / row_cnt) * 100, 2))
# Then filter out only the ones with more than 2 elements
filtered = filtered.where(array_contains(filtered.size, 2))
filtered.show()


# ### Take away
#
# Now we can see that ~35% of respondents answered at least once in the "extreme extrovert" category within the EXT questions, and we didn't have ANY respondents who answered even once in the "extreme introvert" category. That's interesting. I guess can assume that this crowd would have fun at a social gathering at least :)

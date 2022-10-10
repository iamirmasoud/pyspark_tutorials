#!/usr/bin/env python
# coding: utf-8

# # NLP in Pyspark's MLlib
#
# Natural Language Processing (NLP) is a very trendy topic in the data science area today that is really handy for tasks like **chat bots**, movie or **product review analysis** and especially **tweet classification**. In this notebook, we will cover the **classification aspect of NLP** and go over the features that Spark has for cleaning and preparing your data for analysis. We will also touch on how to implement **ML Pipelines** to a few of our data processing steps to help make our code run a bit faster.
#
# As we learned in the NLP concept review lectures, the text you process must first be cleaned, tokenized and vectorized. Essentially, we need to covert our text into a vector of numbers. But how do we do that? Spark has a variety of built in functions to accomplish all of these tasks very easily. We will cover all of it here!
#
# ### Agenda
#
#     1. Review Data (quality check)
#     2. Clean up the data (remove puncuation, special characters, etc.)
#     3. Tokenize text data
#     4. Remove Stopwords
#     5. Zero index our label column
#     5. Create an ML Pipeline (to streamline steps 3-5)
#     6. Vectorize Text column
#          - Count Vectors
#          - TF-IDF
#          - Word2Vec
#     7. Train and Evaluate Model (classification)
#     8. View Predictions

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("NLP").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# **Import Dependencies**

# In[31]:


# For pipeline development
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *  # CountVectorizer,StringIndexer, RegexTokenizer,StopWordsRemover
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import *  # col, udf,regexp_replace,isnull
from pyspark.sql.types import *  # StringType,IntegerType

# ## Read in Dataset
#
# #### Kickstarter Dataset
#
# ##### What is Kickstarter?
# "Kickstarter is an American public-benefit corporation based in Brooklyn, New York, that maintains a global crowdfunding platform, focused on creativity and merchandising. The company's stated mission is to "help bring creative projects to life". Kickstarter, has reportedly received more than $1.9 billion in pledges from 9.4 million backers to fund 257,000 creative projects, such as films, music, stage shows, comics, journalism, video games, technology and food-related projects.
#
# People who back Kickstarter projects are offered tangible rewards or experiences in exchange for their pledges. This model traces its roots to subscription model of arts patronage, where artists would go directly to their audiences to fund their work" ~ Wikipedia
#
# So, what if you can predict if a project will be or not to be able to get the money from their backers?
#
# #### Content
#
# The datastet contains the blurbs or short description of 215,513 projects runned along 2017, all written in english and all labeled with "successful" or "failed", if they get the money or not, respectively. From those texts you can train linguistics models for description, and even embeddings relative to the case.
#
# **Source:** https://www.kaggle.com/oscarvilla/kickstarter-nlp

# In[3]:


path = "Datasets/"

# CSV
df = spark.read.csv(path + "kickstarter.csv", inferSchema=True, header=True)


# In[7]:


df.limit(4).toPandas()


# In[8]:


# Let's read a few full blurbs
df.show(4, False)


# We can see from the output above that the blurb text contains a good bit of punctuation and special characters. We'll need to clean that up.

# In[9]:


df.printSchema()


# **See how many rows are in the df?**

# In[10]:


df.count()


# ## How many null values do we have?
#
# Let's use our handy dandy function!

# In[11]:


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


# Not too bad! Less than 1% for the blurb column and about 5% of the state column. Unfortunatly though, we will need each row of data to contain value in both of these columns to conduct our analysis, so let's how many rows that actually effects.

# In[14]:


# Of course you will want to know how many rows that affected before you actually execute it..
og_len = df.count()
drop_len = df.na.drop().count()
print("Total Rows that contain at least one null value:", og_len - drop_len)
print(
    "Percentage of Rows that contain at least one null value:",
    (og_len - drop_len) / og_len,
)


# So dropping all rows that have at least one null value would impact just under 6% of our dataframe. I can live with that, so I'll go ahead and drop them.

# In[15]:


# Drop the null values
# It's only about 6% so that's okay
df = df.dropna()


# In[16]:


# New df row count
df.count()


# If you want to make this notebook run faster, you can slice the df like this...

# In[17]:


# Slice the dataframe down to make this notebook run faster
df = df.limit(400)
print("Sliced row count:", df.count())


# ## Quality Assurance Check (QA)
#
# Let's make sure our dependent variable column is clean before we go any further. This an important step in our analysis.

# In[18]:


# Quick data quality check on the state column....
# This is going to be our category column so it's important
df.groupBy("state").count().orderBy(col("count").desc()).show(truncate=False)


# We can see from the query above that we have some invalid data in the label (state) column. Let's delete those.

# In[19]:


df = df.filter("state IN('successful','failed')")
# Make sure it worked
df.groupBy("state").count().orderBy(col("count").desc()).show(truncate=False)


# In[20]:


# Let's check the quality of the blurbs
df.select("blurb").show(10, False)


# We see some punctuation proper casing and some slashes which might making parsing problematic. Let's clean this up a bit!

# ## Clean the blurb column
#
# Keep in mind that you can/should do all of this in one call...
# But we will show each individually for the purpose of learning.

# In[21]:


# Replace Slashes and parenthesis with spaces
# You can test your script on line 7 of the df "(Legend of Zelda/Fable Inspired)"
df = (
    df.withColumn("blurb", translate(col("blurb"), "/", " "))
    .withColumn("blurb", translate(col("blurb"), "(", " "))
    .withColumn("blurb", translate(col("blurb"), ")", " "))
)
df.select("blurb").show(7, False)


# In[22]:


# Removing anything that is not a letter
df = df.withColumn("blurb", regexp_replace(col("blurb"), "[^A-Za-z ]+", ""))
df.select("blurb").show(10, False)


# In[23]:


# Remove multiple spaces
df = df.withColumn("blurb", regexp_replace(col("blurb"), " +", " "))
df.select("blurb").show(4, False)


# In[24]:


# Lower case everything
df = df.withColumn("blurb", lower(col("blurb")))
df.select("blurb").show(4, False)


# Take a pause here and go look at your Spark UI. You'll notice that only the "show strings" calls (as opposed to each of the data manipulation calls) are creating jobs. This is because of Sparks lazy computation.
#
# <br>
# So when you want to speed up your notebook those are some calls you can take out.

# ## Prep Data for NLP
#
# Alright so here is where our analysis turns from basic text cleaning to actually turning our text into number (the backbone of NLP). These next several steps in our analysis are very unique to NLP.

# ### Split text into words (Tokenizing)
#
# Yo'll see a new column is added to our dataframe that we call "words". This column contains an array of strings as opposed to just a string (current data type of the blurb column).

# In[25]:


regex_tokenizer = RegexTokenizer(inputCol="blurb", outputCol="words", pattern="\\W")
raw_words = regex_tokenizer.transform(df)
raw_words.show(2, False)
raw_words.printSchema()


# ### Removing Stopwords
#
# **Recall from the content review lecture**
# Recall that "stopwords" are any word that we feel would "distract" our model from performing it's best. This list can be customized, but for now, we will just use the default list.

# In[26]:


# from pyspark.ml.feature import StopWordsRemover

# Define a list of stop words or use default list
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
stopwords = remover.getStopWords()

# Display default list
stopwords[:10]


# In[27]:


words_df = remover.transform(raw_words)
words_df.show(1, False)


# ### Now we need to encode state column to a column of indices
#
# Remember that MLlib requres our dependent variable to not only be a numeric data type, but also zero indexed. We can Sparks handy built in StringIndexer function to accomplish this, just like we did in the classification lectures.

# In[28]:


indexer = StringIndexer(inputCol="state", outputCol="label")
feature_data = indexer.fit(words_df).transform(words_df)
feature_data.show(5)
feature_data.printSchema()


# ## Creat an ML Pipeline
#
# We could also create an ML Pipeline to accomplish the previous three steps in a more streamlined fashion. Pipelines allow users to combine any transformer call(s) and ONE estimator call in their ML workflow. Si a Pipeline can be a continuous set of transformer calls until you reach a point where you need to call ".fit()" which is an estimator call.
# <br>
#
# Notice in the script below that we reduced our .transform calls from 3 to 1. So the benefit here is not necessarily speed but a bit less and more organized code (always nice) and little more streamlined. This feature can be esspecially useful when you get to the point where you want to move your model into production. You can save this pipeline to be called on whenever you need to prep new text.

# In[ ]:


# from pyspark.sql.functions import *
# from pyspark.ml.feature import StopWordsRemover


# In[29]:


######################## BEFORE #############################
# Tokenize
regex_tokenizer = RegexTokenizer(inputCol="blurb", outputCol="words", pattern="\\W")
raw_words = regex_tokenizer.transform(df)

# Remove Stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
words_df = remover.transform(raw_words)

# Zero Index Label Column
indexer = StringIndexer(inputCol="state", outputCol="label")
feature_data = indexer.fit(words_df).transform(words_df)

feature_data.show(1, False)


# In[30]:


################# AFTER ##################

# Tokenize
regex_tokenizer = RegexTokenizer(inputCol="blurb", outputCol="words", pattern="\\W")
# raw_words = regex_tokenizer.transform(df)

# Remove Stop words
remover = StopWordsRemover(
    inputCol=regex_tokenizer.getOutputCol(), outputCol="filtered"
)
# words_df = remover.transform(raw_words)

# Zero Index Label Column
indexer = StringIndexer(inputCol="state", outputCol="label")
# feature_data = indexer.fit(words_df).transform(words_df)

# Create the Pipeline
pipeline = Pipeline(stages=[regex_tokenizer, remover, indexer])
data_prep_pl = pipeline.fit(df)
# print(type(data_prep_pl))
# print(" ")
# Now call on the Pipeline to get our final df
feature_data = data_prep_pl.transform(df)
feature_data.show(1, False)


# Now take a look at the Spark UI again. You'll see the last 2 "countbyvalue" job ids for each one of these. If you take a look at how long it took each of those job ids to run, you will see that the second job id actually took just a bit less time to run. Since we do not have much data here it only saved us .2 seconds but that may translate to a couple of miniutes on a much larger df.

# ## Converting text into vectors
#
# We will test out the following three vectorizors:
#
# 1. Count Vectors
# 2. TF-IDF
# 3. Word2Vec

# In[44]:


# Count Vector (count vectorizer and hashingTF are basically the same thing)
# cv = CountVectorizer(inputCol="filtered", outputCol="features")
# model = cv.fit(feature_data)
# countVectorizer_features = model.transform(feature_data)

# Hashing TF
hashingTF = HashingTF(inputCol="filtered", outputCol="rawfeatures", numFeatures=20)
HTFfeaturizedData = hashingTF.transform(feature_data)

# TF-IDF
idf = IDF(inputCol="rawfeatures", outputCol="features")
idfModel = idf.fit(HTFfeaturizedData)
TFIDFfeaturizedData = idfModel.transform(HTFfeaturizedData)
TFIDFfeaturizedData.name = "TFIDFfeaturizedData"

# rename the HTF features to features to be consistent
HTFfeaturizedData = HTFfeaturizedData.withColumnRenamed("rawfeatures", "features")
HTFfeaturizedData.name = "HTFfeaturizedData"  # We will use later for printing


# In[45]:


# Word2Vec
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="filtered", outputCol="features")
model = word2Vec.fit(feature_data)

W2VfeaturizedData = model.transform(feature_data)
# W2VfeaturizedData.show(1,False)

# W2Vec Dataframes typically has negative values so we will correct for that here so that we can use the Naive Bayes classifier
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(W2VfeaturizedData)

# rescale each feature to range [min, max].
scaled_data = scalerModel.transform(W2VfeaturizedData)
W2VfeaturizedData = scaled_data.select("state", "blurb", "label", "scaledFeatures")
W2VfeaturizedData = W2VfeaturizedData.withColumnRenamed("scaledFeatures", "features")

W2VfeaturizedData.name = "W2VfeaturizedData"  # We will need this to print later


# ## Train and Evaluate your model
#
# From here on out, is straight up classification. So we can go and use our trusty function! I'll just go ahead and copy and paste it in here.

# In[46]:


def ClassTrainEval(classifier, features, classes, train, test):
    def FindMtype(classifier):
        # Intstantiate Model
        M = classifier
        # Learn what it is
        Mtype = type(M).__name__

        return Mtype

    Mtype = FindMtype(classifier)

    def IntanceFitModel(Mtype, classifier, classes, features, train):

        if Mtype == "OneVsRest":
            # instantiate the base classifier.
            lr = LogisticRegression()
            # instantiate the One Vs Rest Classifier.
            OVRclassifier = OneVsRest(classifier=lr)
            #             fitModel = OVRclassifier.fit(train)
            # Add parameters of your choice here:
            paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
            # Cross Validator requires the following parameters:
            crossval = CrossValidator(
                estimator=OVRclassifier,
                estimatorParamMaps=paramGrid,
                evaluator=MulticlassClassificationEvaluator(),
                numFolds=2,
            )  # 3 is best practice
            # Run cross-validation, and choose the best set of parameters.
            fitModel = crossval.fit(train)
            return fitModel
        if Mtype == "MultilayerPerceptronClassifier":
            # specify layers for the neural network:
            # input layer of size features, two intermediate of features+1 and same size as features
            # and output of size number of classes
            # Note: crossvalidator cannot be used here
            features_count = len(features[0][0])
            layers = [features_count, features_count + 1, features_count, classes]
            MPC_classifier = MultilayerPerceptronClassifier(
                maxIter=100, layers=layers, blockSize=128, seed=1234
            )
            fitModel = MPC_classifier.fit(train)
            return fitModel
        if (
            Mtype in ("LinearSVC", "GBTClassifier") and classes != 2
        ):  # These classifiers currently only accept binary classification
            print(
                Mtype,
                " could not be used because PySpark currently only accepts binary classification data for this algorithm",
            )
            return
        if Mtype in (
            "LogisticRegression",
            "NaiveBayes",
            "RandomForestClassifier",
            "GBTClassifier",
            "LinearSVC",
            "DecisionTreeClassifier",
        ):

            # Add parameters of your choice here:
            if Mtype in ("LogisticRegression"):
                paramGrid = (
                    ParamGridBuilder()  #                              .addGrid(classifier.regParam, [0.1, 0.01]) \
                    .addGrid(classifier.maxIter, [10, 15, 20])
                    .build()
                )

            # Add parameters of your choice here:
            if Mtype in ("NaiveBayes"):
                paramGrid = (
                    ParamGridBuilder()
                    .addGrid(classifier.smoothing, [0.0, 0.2, 0.4, 0.6])
                    .build()
                )

            # Add parameters of your choice here:
            if Mtype in ("RandomForestClassifier"):
                paramGrid = (
                    ParamGridBuilder().addGrid(classifier.maxDepth, [2, 5, 10])
                    #                                .addGrid(classifier.maxBins, [5, 10, 20])
                    #                                .addGrid(classifier.numTrees, [5, 20, 50])
                    .build()
                )

            # Add parameters of your choice here:
            if Mtype in ("GBTClassifier"):
                paramGrid = (
                    ParamGridBuilder()  #                              .addGrid(classifier.maxDepth, [2, 5, 10, 20, 30]) \
                    #                              .addGrid(classifier.maxBins, [10, 20, 40, 80, 100]) \
                    .addGrid(classifier.maxIter, [10, 15, 50, 100]).build()
                )

            # Add parameters of your choice here:
            if Mtype in ("LinearSVC"):
                paramGrid = (
                    ParamGridBuilder()
                    .addGrid(classifier.maxIter, [10, 15])
                    .addGrid(classifier.regParam, [0.1, 0.01])
                    .build()
                )

            # Add parameters of your choice here:
            if Mtype in ("DecisionTreeClassifier"):
                paramGrid = (
                    ParamGridBuilder()  #                              .addGrid(classifier.maxDepth, [2, 5, 10, 20, 30]) \
                    .addGrid(classifier.maxBins, [10, 20, 40, 80, 100])
                    .build()
                )

            # Cross Validator requires all of the following parameters:
            crossval = CrossValidator(
                estimator=classifier,
                estimatorParamMaps=paramGrid,
                evaluator=MulticlassClassificationEvaluator(),
                numFolds=2,
            )  # 3 + is best practice
            # Fit Model: Run cross-validation, and choose the best set of parameters.
            fitModel = crossval.fit(train)
            return fitModel

    fitModel = IntanceFitModel(Mtype, classifier, classes, features, train)

    # Print feature selection metrics
    if fitModel is not None:

        if Mtype in ("OneVsRest"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print("\033[1m" + Mtype + "\033[0m")
            # Extract list of binary models
            models = BestModel.models
            for model in models:
                print(
                    "\033[1m" + "Intercept: " + "\033[0m",
                    model.intercept,
                    "\033[1m" + "\nCoefficients:" + "\033[0m",
                    model.coefficients,
                )

        if Mtype == "MultilayerPerceptronClassifier":
            print("")
            print("\033[1m" + Mtype, " Weights" + "\033[0m")
            print("\033[1m" + "Model Weights: " + "\033[0m", fitModel.weights.size)
            print("")

        if Mtype in (
            "DecisionTreeClassifier",
            "GBTClassifier",
            "RandomForestClassifier",
        ):
            # FEATURE IMPORTANCES
            # Estimate of the importance of each feature.
            # Each feature’s importance is the average of its importance across all trees
            # in the ensemble The importance vector is normalized to sum to 1.
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print("\033[1m" + Mtype, " Feature Importances" + "\033[0m")
            print("(Scores add up to 1)")
            print("Lowest score is the least important")
            print(" ")
            print(BestModel.featureImportances)

            if Mtype in ("DecisionTreeClassifier"):
                global DT_featureimportances
                DT_featureimportances = BestModel.featureImportances.toArray()
                global DT_BestModel
                DT_BestModel = BestModel
            if Mtype in ("GBTClassifier"):
                global GBT_featureimportances
                GBT_featureimportances = BestModel.featureImportances.toArray()
                global GBT_BestModel
                GBT_BestModel = BestModel
            if Mtype in ("RandomForestClassifier"):
                global RF_featureimportances
                RF_featureimportances = BestModel.featureImportances.toArray()
                global RF_BestModel
                RF_BestModel = BestModel

        if Mtype in ("LogisticRegression"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print("\033[1m" + Mtype, " Coefficient Matrix" + "\033[0m")
            print("You should compares these relative to eachother")
            print("Coefficients: \n" + str(BestModel.coefficientMatrix))
            print("Intercept: " + str(BestModel.interceptVector))
            global LR_coefficients
            LR_coefficients = BestModel.coefficientMatrix.toArray()
            global LR_BestModel
            LR_BestModel = BestModel

        if Mtype in ("LinearSVC"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print("\033[1m" + Mtype, " Coefficients" + "\033[0m")
            print("You should compares these relative to eachother")
            print("Coefficients: \n" + str(BestModel.coefficients))
            global LSVC_coefficients
            LSVC_coefficients = BestModel.coefficients.toArray()
            global LSVC_BestModel
            LSVC_BestModel = BestModel

    # Set the column names to match the external results dataframe that we will join with later:
    columns = ["Classifier", "Result"]

    if Mtype in ("LinearSVC", "GBTClassifier") and classes != 2:
        Mtype = [Mtype]  # make this a list
        score = ["N/A"]
        result = spark.createDataFrame(zip(Mtype, score), schema=columns)
    else:
        predictions = fitModel.transform(test)
        MC_evaluator = MulticlassClassificationEvaluator(
            metricName="accuracy"
        )  # redictionCol="prediction",
        accuracy = (MC_evaluator.evaluate(predictions)) * 100
        Mtype = [Mtype]  # make this a string
        score = [str(accuracy)]  # make this a string and convert to a list
        result = spark.createDataFrame(zip(Mtype, score), schema=columns)
        result = result.withColumn("Result", result.Result.substr(0, 5))

    return result
    # Also returns the fit model important scores or p values


# Declare the algorithims you want to test plus declare a list of all the different feature vectors we want to test out that we created above.

# In[47]:


# from pyspark.ml.classification import *
# from pyspark.ml.evaluation import *
# from pyspark.sql import functions
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Comment out Naive Bayes if your data still contains negative values
classifiers = [
    LogisticRegression(),
    OneVsRest(),
    LinearSVC(),
    NaiveBayes(),
    RandomForestClassifier(),
    GBTClassifier(),
    DecisionTreeClassifier(),
    MultilayerPerceptronClassifier(),
]

featureDF_list = [HTFfeaturizedData, TFIDFfeaturizedData, W2VfeaturizedData]


# Loop through all feature types (hashingTF, TFIDF and Word2Vec)

# In[48]:


for featureDF in featureDF_list:
    print("\033[1m" + featureDF.name, " Results:" + "\033[0m")
    train, test = featureDF.randomSplit([0.7, 0.3], seed=11)
    features = featureDF.select(["features"]).collect()
    # Learn how many classes there are in order to specify evaluation type based on binary or multi and turn the df into an object
    class_count = featureDF.select(countDistinct("label")).collect()
    classes = class_count[0][0]

    # set up your results table
    columns = ["Classifier", "Result"]
    vals = [("Place Holder", "N/A")]
    results = spark.createDataFrame(vals, columns)

    for classifier in classifiers:
        new_result = ClassTrainEval(classifier, features, classes, train, test)
        results = results.union(new_result)
    results = results.where("Classifier!='Place Holder'")
    print(results.show(truncate=False))


# Looks like the Decision Tree classifier with the W2VfeaturizedData was our best performing feature list/classifier combo. Let's go with that and create our final model and play around with the test dataframe.

# In[49]:


classifier = DecisionTreeClassifier()
featureDF = W2VfeaturizedData

train, test = featureDF.randomSplit([0.7, 0.3], seed=11)
features = featureDF.select(["features"]).collect()

# Learn how many classes there are in order to specify evaluation type based on binary or multi and turn the df into an object
class_count = featureDF.select(countDistinct("label")).collect()
classes = class_count[0][0]

# running this afain with generate all the objects need to play around with test data
ClassTrainEval(classifier, features, classes, train, test)


# Let's see some results!

# In[50]:


predictions = DT_BestModel.transform(test)
print("Predicted Failures:")
predictions.select("state", "blurb").filter("prediction=0").orderBy(
    predictions["prediction"].desc()
).show(3, False)
print(" ")
print("Predicted Success:")
predictions.select("state", "blurb").filter("prediction=1").orderBy(
    predictions["prediction"].desc()
).show(3, False)


# ## What could be next?
#
# Once we have our model and all the vectorizer the sky is really the limit! We could do any of the following for starters:
#
# 1. Allow a user to input their own "blurb" and we could return a prediction of whether or not it would pass
# 2. If we had a time variable here, we could show the most popular words over time
# 3. Provide this algorithim to Kickstarter for prescreening so they can prioritize entries

# In[ ]:

#!/usr/bin/env python
# coding: utf-8

# # NLP in Pyspark's MLlib Project Solution
#
# ## Fake Job Posting Predictions
#
# Indeed.com has just hired you to create a system that automatically flags suspicious job postings on it's website. It has recently seen an influx of fake job postings that is negativley impacting it's customer experience. Becuase of the high volume of job postings it receives everyday, their employees don't have the capacity to check every posting so they would like an automated system that prioritizes which postings to review before deleting it.
#
# #### Your task
# Use the attached dataset to create an NLP alogorthim which automatically flags suspicious posts for review.
#
# #### The data
# This dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs.
#
# **Data Source:** https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
#
# #### My basic approach
# I think I will use just the job description variable (there are a few lengthier text variables we could have chosen from) for my analysis for now and think about the company profile for another analysis later on. Something that would be cool here would be to create multiple models that all work together to provide a recommendation or a score of how likley it is that a job posting is fake.

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


# **Read in dependencies**

# In[2]:


from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.ml import Pipeline
from pyspark.ml.feature import *  # CountVectorizer,StringIndexer, RegexTokenizer,StopWordsRemover
from pyspark.sql.functions import *  # col, udf,regexp_replace,isnull
from pyspark.sql.types import IntegerType, StringType

# **And the dataset**

# In[3]:


path = "Datasets/"

# CSV
postings = spark.read.csv(path + "fake_job_postings.csv", inferSchema=True, header=True)


# **View the data for QA**

# In[4]:


postings.limit(4).toPandas()


# In[5]:


# Let's read a full line of data of fradulent postings
postings.filter("fraudulent=1").show(1, False)
# These look good!


# In[6]:


postings.printSchema()


# **See how many rows are in the df**

# In[7]:


postings.count()


# ## Null Values?

# In[8]:


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


null_columns_calc_list = null_value_calc(postings)
spark.createDataFrame(
    null_columns_calc_list, ["Column_Name", "Null_Values_Count", "Null_Value_Percent"]
).show()


# Quite a bit of missing data here. We better be careful dropping

# In[9]:


# Let's see how much total
og_len = postings.count()
drop_len = postings.na.drop().count()
print("Total Null Rows:", og_len - drop_len)
print("Percentage Null Rows", (og_len - drop_len) / og_len)


# Wawwww 95% is wayyyy too much. Better find a better approach.

# In[10]:


# How about by subset by just the vars we need for now.
df = postings.na.drop(subset=["fraudulent", "description"])


# **Much better**

# In[11]:


df.count()


# In[12]:


# Quick data quality check on the dependent var....
# This should be a binary outcome (0 or 1)
df.groupBy("fraudulent").count().orderBy(col("count").desc()).show(8)


# We can see from the query above that we have some invalid data in the label (fraudulent) column. Let's delete those.

# In[13]:


df = df.filter("fraudulent IN('0','1')")
# QA again
df.groupBy("fraudulent").count().show(truncate=False)


# ### Balance the signal
#
# The other thing I want to do is resample the dataframe so I get a better signal from the data since there is not many fraudulent cases. I mentioned class imbalance earlier on, but we haven't come accross a good example yet. This is a good one where we see that the ratio between the fraudent cases and the real cases is extremley unbalanced. So undersampling the non-fradulent cases will help with that.
#
# Luckily, Spark has a cool built in function for this called sampleby to accomplish this.

# In[14]:


df = df.sampleBy("fraudulent", fractions={"0": 0.4, "1": 1.0}, seed=10)
# QA again
df.groupBy("fraudulent").count().show(truncate=False)


# That's better!

# ### Encode the label column

# In[15]:


# Let's go ahead and encode it too
indexer = StringIndexer(inputCol="fraudulent", outputCol="label")
df = indexer.fit(df).transform(df)
df.limit(6).toPandas()


# In[16]:


# Let's check the quality of the description var
df.select("description").show(1, False)


# Looks pretty standard.

# ## Clean the datasets

# In[17]:


# Removing anything that is not a letter
df = df.withColumn("description", regexp_replace(df["description"], "[^A-Za-z ]+", ""))
# Remove multiple spaces
df = df.withColumn("description", regexp_replace(df["description"], " +", " "))
# Lower case everything
df = df.withColumn("description", lower(df["description"]))


# In[18]:


df.limit(5).toPandas()


# ## Split text into words (Tokenizing)

# In[19]:


regex_tokenizer = RegexTokenizer(
    inputCol="description", outputCol="words", pattern="\\W"
)
df = regex_tokenizer.transform(df)

df.limit(5).toPandas()


# ## Removing Stopwords

# In[20]:


from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
feature_data = remover.transform(df)

feature_data.limit(5).toPandas()


# ## Converting text into vectors
#
# We test out the following three vectors
#
# 1. Count Vectors
# 2. TF-IDF
# 3. Word2Vec

# In[21]:


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


# In[22]:


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
W2VfeaturizedData = scaled_data.select(
    "fraudulent", "description", "label", "scaledFeatures"
)
W2VfeaturizedData = W2VfeaturizedData.withColumnRenamed("scaledFeatures", "features")

W2VfeaturizedData.name = "W2VfeaturizedData"  # We will need this to print later


# ## Train and Evaluate your model
#
# From here on out, is straight up classification. So we can go and use our trusty function!

# In[23]:


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


# Read in all dependencies and declare the algorithims you want to test

# In[24]:


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

# In[25]:


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


# Looks like the Random Forest classifier with either the HTFfeaturizedData or the TFIDFfeaturizedData are our best performing feature list/classifier combos. Let's go with the Hashing TF vector for the sake of simiplicity and create our final model and play around with the test dataframe.

# In[26]:


# Train final model
classifier = RandomForestClassifier()
featureDF = HTFfeaturizedData

train, test = featureDF.randomSplit([0.7, 0.3], seed=11)
features = featureDF.select(["features"]).collect()

# Learn how many classes there are in order to specify evaluation type based on binary or multi and turn the df into an object
class_count = featureDF.select(countDistinct("label")).collect()
classes = class_count[0][0]

# running this afain with generate all the objects need to play around with test data
ClassTrainEval(classifier, features, classes, train, test)


# Let's see some results!

# In[27]:


predictions = RF_BestModel.transform(test)
print("Predicted Fraudulent:")
predictions.select("fraudulent", "description").filter("prediction=1").orderBy(
    predictions["prediction"].desc()
).show(3, False)
print(" ")
print("Predicted Not Fraudulent:")
predictions.select("fraudulent", "description").filter("prediction=0").orderBy(
    predictions["prediction"].desc()
).show(3, False)


# ## What could be next?
#
# This analysis was really just the tip of the ice berg here. We could also consider the following analysis:
#
# 1. Autotag suspicious descriptions
# 2. Conduct a similar analysis on the company profile field and the requirements field (NLP)
# 3. Frequent pattern mining on null and non null values in other fields
# 4. Consider doing anlaysis on amount of typos in the description

# In[ ]:

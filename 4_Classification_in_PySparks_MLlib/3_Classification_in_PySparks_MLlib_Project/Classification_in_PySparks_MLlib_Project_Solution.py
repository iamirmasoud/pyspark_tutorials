#!/usr/bin/env python
# coding: utf-8

# # Classification in PySpark's MLlib Project Solution
#
# ### Genre classification
# Now it's time to leverage what we learned in the lectures to a REAL classification project! Have you ever wondered what makes us, humans, able to tell apart two songs of different genres? How we do we inherenly know the difference between a pop song and heavy metal? This type of classifcation may seem easy for us, but it's a very difficult challenge for a computer to do. So the question is, could an automatic genre classifcation model be possible?
#
# For this project we will be classifying songs based on a number of characteristics into a set of 23 electronic genres. This technology could be used by an application like Pandora to recommend songs to users or just create meaningful channels. Super fun!
#
# ### Dataset
# *beatsdataset.csv*
# Each row is an electronic music song. The dataset contains 100 song for each genre among 23 electronic music genres, they were the top (100) songs of their genres on November 2016. The 71 columns are audio features extracted of a two random minutes sample of the file audio. These features have been extracted using pyAudioAnalysis (https://github.com/tyiannak/pyAudioAnalysis).
#
# ### Your task
# Create an algorithm that classifies songs into the 23 genres provided. Test out several different models and select the highest performing one. Also play around with feature selection methods and finally try to make a recommendation to a user.
#
# For the feature selection aspect of this project, you may need to get a bit creative if you want to select features from a non-tree algorithm. I did not go over this aspect of PySpark intentionally in the previous lectures to give you chance to get used to researching the PySpark documentation page. Here is the link to the Feature Selectors section of the documentation that just might come in handy: https://spark.apache.org/docs/latest/ml-features.html#feature-selectors
#
# Good luck! Have fun :)
#
#
# ### My approach
#
# I decided to approach this analysis in 4 main steps.
#
# 1. **Create Baseline:** Train and evaluate models on raw data without pre-treating it for outliers, skewness or negative values. This way we can clearly see what effect our transformations have on our analysis.
#
# 2. **Test treatments:** Train and evaluate models on treated data (outliers, skewness and negative values) and compare to baseline (#1).
#
# 3. **Feature Selection:** Select the best performing models from the previous two approaches and perform feature selection on it to fine tune it.
#
# 4. **Make a recommendation to a user:** Create a scrip to make a recommendation to a user. I intentionally left this part of the project a bit ambiguous
#
# ### Source
# https://www.kaggle.com/caparrini/beatsdataset

# First things first... let's create our PySpark instance.

# In[3]:


# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("ClassificationPS").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# In[4]:


from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
# Read in dependencies
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Read in our dataset

# In[5]:


path = "Datasets/"
df = spark.read.csv(path + "beatsdataset.csv", inferSchema=True, header=True)


# ### Check out the dataset
#
# Let's produce a print out of the dataframe so we know what we are working with.

# In[4]:


df.limit(6).toPandas()


# In[5]:


df.printSchema()


# ### How many classes do we have and are they balanced?
#
# Just making sure :)
#
# We have a perfectly balanced dataset, however we only have 100 examples from each class which may make training a decent model challenging. But let's see what we can do!
#
# *Note: This almost never happens in real life :)*

# In[6]:


df.groupBy("class").count().show(100)


# ## Set up our Data Formatting Function
#
# Remember that MLlib requires all input columns of your dataframe to be vectorized and our dependent variable needs to be zero indexed. We can do that using our handy dandy function we developed in the lecture. Feel free to make this your own!
#
# For example, with this go, I added a print statement to show the new label values for each class.

# In[6]:


# Data Prep function
def MLClassifierDFPrep(
    df, input_columns, dependent_var, treat_outliers=True, treat_neg_values=True
):

    # change label (class variable) to string type to prep for reindexing
    # Pyspark is expecting a zero indexed integer for the label column.
    # Just incase our data is not in that format... we will treat it by using the StringIndexer built in method
    renamed = df.withColumn(
        "label_str", df[dependent_var].cast(StringType())
    )  # Rename and change to string type
    indexer = StringIndexer(
        inputCol="label_str", outputCol="label"
    )  # Pyspark is expecting the this naming convention
    indexed = indexer.fit(renamed).transform(renamed)
    print(indexed.groupBy("class", "label").count().show(100))

    # Convert all string type data in the input column list to numeric
    # Otherwise the Algorithm will not be able to process it
    numeric_inputs = []
    string_inputs = []
    for column in input_columns:
        if str(indexed.schema[column].dataType) == "StringType":
            indexer = StringIndexer(inputCol=column, outputCol=column + "_num")
            indexed = indexer.fit(indexed).transform(indexed)
            new_col_name = column + "_num"
            string_inputs.append(new_col_name)
        else:
            numeric_inputs.append(column)

    if treat_outliers == True:
        print("We are correcting for non normality now!")
        # empty dictionary d
        d = {}
        # Create a dictionary of quantiles
        for col in numeric_inputs:
            d[col] = indexed.approxQuantile(
                col, [0.01, 0.99], 0.25
            )  # if you want to make it go faster increase the last number
        # Now fill in the values
        for col in numeric_inputs:
            skew = indexed.agg(skewness(indexed[col])).collect()  # check for skewness
            skew = skew[0][0]
            # This function will floor, cap and then log+1 (just in case there are 0 values)
            if skew > 1:
                indexed = indexed.withColumn(
                    col,
                    log(
                        when(df[col] < d[col][0], d[col][0])
                        .when(indexed[col] > d[col][1], d[col][1])
                        .otherwise(indexed[col])
                        + 1
                    ).alias(col),
                )
                print(
                    col + " has been treated for positive (right) skewness. (skew =)",
                    skew,
                    ")",
                )
            elif skew < -1:
                indexed = indexed.withColumn(
                    col,
                    exp(
                        when(df[col] < d[col][0], d[col][0])
                        .when(indexed[col] > d[col][1], d[col][1])
                        .otherwise(indexed[col])
                    ).alias(col),
                )
                print(
                    col + " has been treated for negative (left) skewness. (skew =",
                    skew,
                    ")",
                )

    # Produce a warning if there are negative values in the dataframe that Naive Bayes cannot be used.
    # Note: we only need to check the numeric input values since anything that is indexed won't have negative values
    minimums = df.select(
        [min(c).alias(c) for c in df.columns if c in numeric_inputs]
    )  # Calculate the mins for all columns in the df
    min_array = minimums.select(
        array(numeric_inputs).alias("mins")
    )  # Create an array for all mins and select only the input cols
    df_minimum = min_array.select(
        array_min(min_array.mins)
    ).collect()  # Collect golobal min as Python object
    df_minimum = df_minimum[0][0]  # Slice to get the number itself

    features_list = numeric_inputs + string_inputs
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")
    output = assembler.transform(indexed).select("features", "label")

    #     final_data = output.select('features','label') #drop everything else

    # Now check for negative values and ask user if they want to correct that?
    if df_minimum < 0:
        print(" ")
        print(
            "WARNING: The Naive Bayes Classifier will not be able to process your dataframe as it contains negative values"
        )
        print(" ")

    if treat_neg_values == True:
        print(
            "You have opted to correct that by rescaling all your features to a range of 0 to 1"
        )
        print(" ")
        print("We are rescaling you dataframe....")
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

        # Compute summary statistics and generate MinMaxScalerModel
        scalerModel = scaler.fit(output)

        # rescale each feature to range [min, max].
        scaled_data = scalerModel.transform(output)
        final_data = scaled_data.select(
            "label", "scaledFeatures"
        )  # added class to the selection
        final_data = final_data.withColumnRenamed("scaledFeatures", "features")
        print("Done!")

    else:
        print(
            "You have opted not to correct that therefore you will not be able to use to Naive Bayes classifier"
        )
        print("We will return the dataframe unscaled.")
        final_data = output

    return final_data


# ## Set up our Training and Evaluation Function
#
# Let's use our handy dandy function to train and test all our classifiers we have available to us! I made two modifications here to show you just how flexible this framework is.
#
# 1. You'll see in the area where we print our coefficients and feature importances values, that I created a new output format that joins the values with their corresponding feature names so we can clearly see which value belongs to which feature. I did this by zipping the lists together and creating a Spark dataframe to allow the output to be viewed in column row format as opposed to a messy list.
# 2. The second change I added was a parameter for the folds so that I could specify the folds more easily with each iteration.
#
# Let's check it out!
#
# For more info on available hyper parameters visit: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification

# In[7]:


def ClassTrainEval(classifier, features, classes, folds, train, test):
    def FindMtype(classifier):
        # Intstantiate Model
        M = classifier
        # Learn what it is
        Mtype = type(M).__name__

        return Mtype

    Mtype = FindMtype(classifier)

    def IntanceFitModel(Mtype, classifier, classes, features, folds, train):

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
                numFolds=folds,
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
                numFolds=folds,
            )  # 3 + is best practice
            # Fit Model: Run cross-validation, and choose the best set of parameters.
            fitModel = crossval.fit(train)
            return fitModel

    fitModel = IntanceFitModel(Mtype, classifier, classes, features, folds, train)

    # Print feature selection metrics
    if fitModel is not None:

        if Mtype in ("OneVsRest"):
            # Get Best Model
            BestModel = fitModel.bestModel
            global OVR_BestModel
            OVR_BestModel = BestModel
            print(" ")
            print("\033[1m" + Mtype + "\033[0m")
            # Extract list of binary models
            models = BestModel.models
            for model in models:
                print("\033[1m" + "Intercept: " + "\033[0m", model.intercept)
                print("\033[1m" + "Top 20 Coefficients:" + "\033[0m")
                coeff_array = model.coefficients.toArray()
                coeff_scores = []
                for x in coeff_array:
                    coeff_scores.append(float(x))
                # Then zip with input_columns list and create a df
                result = spark.createDataFrame(
                    zip(input_columns, coeff_scores), schema=["feature", "coeff"]
                )
                print(result.orderBy(result["coeff"].desc()).show(truncate=False))

        if Mtype == "MultilayerPerceptronClassifier":
            print("")
            print("\033[1m" + Mtype + "\033[0m")
            print("\033[1m" + "Model Weights: " + "\033[0m", fitModel.weights.size)
            print("")
            global MLPC_Model
            MLPC_BestModel = fitModel

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
            print("\033[1m" + Mtype, " Top 20 Feature Importances" + "\033[0m")
            print("(Scores add up to 1)")
            print("Lowest score is the least important")
            print(" ")
            featureImportances = BestModel.featureImportances.toArray()
            # Convert from numpy array to list
            imp_scores = []
            for x in featureImportances:
                imp_scores.append(float(x))
            # Then zip with input_columns list and create a df
            result = spark.createDataFrame(
                zip(input_columns, imp_scores), schema=["feature", "score"]
            )
            print(result.orderBy(result["score"].desc()).show(truncate=False))

            # Save the feature importance values and the models
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

        # Print the coefficients
        if Mtype in ("LogisticRegression"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print("\033[1m" + Mtype + "\033[0m")
            print("Intercept: " + str(BestModel.interceptVector))
            print("\033[1m" + " Top 20 Coefficients" + "\033[0m")
            print("You should compares these relative to eachother")
            # Convert from numpy array to list
            coeff_array = BestModel.coefficientMatrix.toArray()
            coeff_scores = []
            for x in coeff_array[0]:
                coeff_scores.append(float(x))
            # Then zip with input_columns list and create a df
            result = spark.createDataFrame(
                zip(input_columns, coeff_scores), schema=["feature", "coeff"]
            )
            print(result.orderBy(result["coeff"].desc()).show(truncate=False))
            # Save the coefficient values and the models
            global LR_coefficients
            LR_coefficients = BestModel.coefficientMatrix.toArray()
            global LR_BestModel
            LR_BestModel = BestModel

        # Print the Coefficients
        if Mtype in ("LinearSVC"):
            # Get Best Model
            BestModel = fitModel.bestModel
            print(" ")
            print("\033[1m" + Mtype + "\033[0m")
            print("Intercept: " + str(BestModel.intercept))
            print("\033[1m" + "Top 20 Coefficients" + "\033[0m")
            print("You should compares these relative to eachother")
            #             print("Coefficients: \n" + str(BestModel.coefficients))
            coeff_array = BestModel.coefficients.toArray()
            coeff_scores = []
            for x in coeff_array:
                coeff_scores.append(float(x))
            # Then zip with input_columns list and create a df
            result = spark.createDataFrame(
                zip(input_columns, coeff_scores), schema=["feature", "coeff"]
            )
            print(result.orderBy(result["coeff"].desc()).show(truncate=False))
            # Save the coefficient values and the models
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


# ## Testing Time!
#
# Now we can use these functions above to train and evaluate our models in different ways. I'll show you my approach here but everyone has their own style and this process will likley be much more involved in a real use case, but I just want to show you the art of the possible here quickly.

# In[8]:


# Set up independ and dependent vars
input_columns = df.columns
input_columns = input_columns[
    1:-1
]  # keep only relevant columns: everything but the first and last cols
dependent_var = "class"

# Learn how many classes there are in order to specify evaluation type based on binary or multi and turn the df into an object
class_count = df.select(countDistinct("class")).collect()
classes = class_count[0][0]


# ### Test 1: Without outlier treatment, skew or negative value treatment
#
# This first go will act as our baseline so we can understand how our treatments affected our analysis. So I will opt out of the option for outlier treatment, skew treatment or negative value treatment which means we cannot use the naive bayes classifier.

# In[12]:


# Call on data prep, train and evaluate functions
test1_data = MLClassifierDFPrep(
    df, input_columns, dependent_var, treat_outliers=False, treat_neg_values=False
)
test1_data.limit(5).toPandas()

# Comment out Naive Bayes if your data still contains negative values
classifiers = [
    LogisticRegression(),
    OneVsRest(),
    LinearSVC()
    #                ,NaiveBayes()
    ,
    RandomForestClassifier(),
    GBTClassifier(),
    DecisionTreeClassifier(),
    MultilayerPerceptronClassifier(),
]

train, test = test1_data.randomSplit([0.7, 0.3])
features = test1_data.select(["features"]).collect()
folds = 2  # because we have limited data

# set up your results table
columns = ["Classifier", "Result"]
vals = [("Place Holder", "N/A")]
results = spark.createDataFrame(vals, columns)

for classifier in classifiers:
    new_result = ClassTrainEval(classifier, features, classes, folds, train, test)
    results = results.union(new_result)
results = results.where("Classifier!='Place Holder'")
print("!!!!!Final Results!!!!!!!!")
results.show(100, False)


# ### Test 2: Test treatments
#
# Train and evaluate models on treated data (outliers, skewness and negative values) and compare to baseline (#1).

# In[9]:


# Call on data prep, train and evaluate functions
test2_data = MLClassifierDFPrep(
    df, input_columns, dependent_var, treat_outliers=True, treat_neg_values=True
)
test2_data.limit(5).toPandas()

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

train, test = test2_data.randomSplit([0.7, 0.3])
features = test2_data.select(["features"]).collect()
folds = 2

# set up your results table
columns = ["Classifier", "Result"]
vals = [("Place Holder", "N/A")]
results = spark.createDataFrame(vals, columns)

for classifier in classifiers:
    new_result = ClassTrainEval(classifier, features, classes, folds, train, test)
    results = results.union(new_result)
results = results.where("Classifier!='Place Holder'")
print("!!!!!Final Results!!!!!!!!")
results.show(100, False)


# ## Test 3: Feature Selection
#
# Looks like all our of our models saw a little bit of improvement from the transformations we applied. And overall the best performing model was the One vs Rest. Let's try to do do some feature selection to make it even better!
#
# Since we are using the One vs Rest algorithm backed by Logistic Regression, we can use Pyspark's Chi Squared Selector feature to select our features. I intentionally did not go over this in the previous lectures to get you used to exploring the PySpark documentation.
#
# Here is the link to the documentation on the Chi Squared Selector for more details if you need it: https://spark.apache.org/docs/latest/ml-features#chisqselector

# In[ ]:


from pyspark.ml.feature import ChiSqSelector, VectorSlicer
from pyspark.ml.linalg import Vectors

classifiers = [OneVsRest()]

# Select the top n features and view results
maximum = len(input_columns)
for n in range(10, maximum, 10):
    print("Testing top n = ", n, " features")

    # For Tree classifiers
    #     best_n_features = RF_featureimportances.argsort()[-n:][::-1]
    #     best_n_features= best_n_features.tolist() # convert to a list
    #     vs = VectorSlicer(inputCol="features", outputCol="best_features", indices=best_n_features)
    #     bestFeaturesDf = vs.transform(test2_data)

    # For Logistic regression or One vs Rest
    selector = ChiSqSelector(
        numTopFeatures=n,
        featuresCol="features",
        outputCol="selectedFeatures",
        labelCol="label",
    )
    bestFeaturesDf = selector.fit(test2_data).transform(test2_data)
    bestFeaturesDf = bestFeaturesDf.select("label", "selectedFeatures")
    bestFeaturesDf = bestFeaturesDf.withColumnRenamed("selectedFeatures", "features")

    # Collect features
    features = bestFeaturesDf.select(["features"]).collect()

    # Split
    train, test = bestFeaturesDf.randomSplit([0.7, 0.3])

    # Specify folds
    folds = 2

    # set up your results table
    columns = ["Classifier", "Result"]
    vals = [("Place Holder", "N/A")]
    results = spark.createDataFrame(vals, columns)

    for classifier in classifiers:
        new_result = ClassTrainEval(classifier, features, classes, folds, train, test)
        results = results.union(new_result)
    results = results.where("Classifier!='Place Holder'")
    results.show(100, False)


# # Results
#
# Let's see if we can improve upon the results above of 45.94%.
#
# - 10 features = 32.27% (going down)
# - 20 features = 39.23% (going down)
# - 30 features = 39.24% (going down)
# - 40 features = 42.61% (going down)
# - 50 features = 43.52% (going down)
# - 60 features = 42.42% (going down)
# - 70 features = 40.22% (going down)
#
# **Take away:** looks like we actually never beat our baseline so let's stick with the 71 features. Setting a random seed might have helped with variability we saw here.
#
# ## Train final model
#
# Now we can finally train our final model with the optimal set of features!

# In[ ]:


from pyspark.ml.feature import ChiSqSelector, VectorSlicer
from pyspark.ml.linalg import Vectors

classifiers = [OneVsRest()]

# Select the top n features and view results
n = 71

# For Logistic regression or One vs Rest
selector = ChiSqSelector(
    numTopFeatures=n,
    featuresCol="features",
    outputCol="selectedFeatures",
    labelCol="label",
)
bestFeaturesDf = selector.fit(test2_data).transform(test2_data)
bestFeaturesDf = bestFeaturesDf.select("label", "selectedFeatures")
bestFeaturesDf = bestFeaturesDf.withColumnRenamed("selectedFeatures", "features")

# Collect features
features = bestFeaturesDf.select(["features"]).collect()

# Split
train, test = bestFeaturesDf.randomSplit([0.7, 0.3])

# Specify folds
folds = 2

# set up your results table
columns = ["Classifier", "Result"]
vals = [("Place Holder", "N/A")]
results = spark.createDataFrame(vals, columns)

for classifier in classifiers:
    new_result = ClassTrainEval(classifier, features, classes, folds, train, test)
    results = results.union(new_result)
results = results.where("Classifier!='Place Holder'")
results.show(100, False)


# ## Make a recommendation to a user!
#
# Imagine that your new model is being deployed in the next version release of the online radio station that has now purchased this model from you. So the existing users have gotten used to the songs in certian stations but now your model may be adding new songs to those stations.
#
# Can you show an output that lists songs that will be new to listers of the BigRoom station (ie. songs that used to belong to another station, but are now classified by your model as being in the BigRoom station)? Also show how many songs this would include. You can use the test dataframe to do this. Don't over think it.

# In[ ]:


predictions = OVR_BestModel.transform(test)


# In[ ]:


# From the output earlier we saw that the new label for BigRoom is now 21.0
# Let's get a song from there
count = predictions.filter("label!=21.0 AND prediction == 21.0").count()
print(count)
predictions.filter("label!=21.0 AND prediction == 21.0").show()
# predictions.show()


# ## Conclusions
#
# #### Other types of analyses to consider
#
# 1. **Training and evaluation splits::** we did a 70/30 split but it's a good idea to play around with other split ratios like 80/20 or 75/25
# 2. **Different transformations:** you could also play around with other types of tranformations on the data like normalizing or standard scaling:
#         - normalizing: https://spark.apache.org/docs/2.1.0/ml-features.html#normalizer
#         - standard scaling: https://spark.apache.org/docs/2.1.0/ml-features.html#standardscaler

#!/usr/bin/env python
# coding: utf-8

# # Classification in PySpark's MLlib
#
# PySpark offers a good variety of algorithms that can be applied to classification machine learning problems. However, because PySpark operates on distributed dataframes, we cannot use popular Python libraries like scikit learn for our machine learning applications. Which means we need to use PySpark's MLlib packages for these tasks. Luckily, MLlib offers a pretty good variety of algorithms! In this notebook we will go over how to prep our data, how to train and test the classification algorithms PySpark offers, and also some pretty neat functions that I created to make the model selection process really efficient.
#
# Cover basic structure, training and validation split, model selection, pipelines, Cross validation
#
# ## Defining Classification
#
# In machine learning and statistics, classification is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known (ie. supervised learning). You can think of classification like bucketing.
#
# You should use classification if your **dependent variable** (the variable you want to predict) is a **descrete value** (i.e. whole numbers like 1,2,3,4) typically the dependent variable will represent some class or group like types of flowers or colors, as opposed to a continuous variable like dollar amount, age or blood pressure.
#
# ## Algorithms Available
#
# PySpark offers the following algorithms for classification.
#
# 1. Logistic Regression
# 2. Naive Bayes
# 3. One Vs Rest
# 4. Linear Support Vector Machine (SVC)
# 5. Random Forest Classifier
# 6. GBT Classifier
# 7. Decision Tree Classifier
# 8. Multilayer Perceptron Classifier (Neural Network)

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Review2").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark
# Click the hyperlinked "Spark UI" link to view details about your Spark session


# ## Let's read our dataset in for this notebook
#
# ### Data Set Name: Autistic Spectrum Disorder Screening Data for Adult
# Autistic Spectrum Disorder (ASD) is a neurodevelopment condition associated with significant healthcare costs, and early diagnosis can significantly reduce these. Unfortunately, waiting times for an ASD diagnosis are lengthy and procedures are not cost effective. The economic impact of autism and the increase in the number of ASD cases across the world reveals an urgent need for the development of easily implemented and effective screening methods. Therefore, a time-efficient and accessible ASD screening is imminent to help health professionals and inform individuals whether they should pursue formal clinical diagnosis. The rapid growth in the number of ASD cases worldwide necessitates datasets related to behaviour traits. However, such datasets are rare making it difficult to perform thorough analyses to improve the efficiency, sensitivity, specificity and predictive accuracy of the ASD screening process. Presently, very limited autism datasets associated with clinical or screening are available and most of them are genetic in nature. Hence, we propose a new dataset related to autism screening of adults that contained 20 features to be utilised for further analysis especially in determining influential autistic traits and improving the classification of ASD cases. In this dataset, we record ten behavioural features (AQ-10-Adult) plus ten individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.
#
# ### Source:
# https://www.kaggle.com/faizunnabi/autism-screening

# In[2]:


path = "Datasets/"
df = spark.read.csv(
    path + "Toddler Autism dataset July 2018.csv", inferSchema=True, header=True
)


# ### Check out the dataset

# In[3]:


df.limit(6).toPandas()


# In[4]:


df.printSchema()


# ### How many classes do we have?

# In[5]:


df.groupBy("Class/ASD Traits ").count().show(100)


# ## Format Data
#
# MLlib requires all input columns of your dataframe to be vectorized. You will see that we rename our dependent var to label as that is what is expected for all MLlib applications. If rename once here, we never have to do it again!

# Let's go ahead and create a function to do all of this

# In[3]:


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

    # Convert all string type data in the input column list to numeric
    # Otherwise the Algorithm will not be able to process it
    numeric_inputs = []
    string_inputs = []
    for column in input_columns:
        if indexed.schema[column].dataType == StringType():
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
        final_data = scaled_data.select("label", "scaledFeatures")
        final_data = final_data.withColumnRenamed("scaledFeatures", "features")
        print("Done!")

    else:
        print(
            "You have opted not to correct that therefore you will not be able to use to Naive Bayes classifier"
        )
        print("We will return the dataframe unscaled.")
        final_data = output

    return final_data


# **Take it for a test run!**

# In[4]:


# Read in functions we will need
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

col_list = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "Age_Mons",
    "Qchat-10-Score",
    "Sex",
    "Ethnicity",
    "Jaundice",
    "Family_mem_with_ASD",
    "Who completed the test",
]

# input_columns = df.columns # Collect the column names as a list
# input_columns = input_columns[8:] # keep only relevant columns: from column 8 until the end
input_columns = col_list
dependent_var = "Class/ASD Traits "

final_data = MLClassifierDFPrep(df, input_columns, dependent_var)
final_data.limit(5).toPandas()


# In[8]:


df.show()


# **Split into Test and Training datasets**

# In[5]:


train, test = final_data.randomSplit([0.7, 0.3])


# ## Train!
#
# Let's go ahead and train a Logistic Regression Algorithm

# In[6]:


from pyspark.ml.classification import *

classifier = LogisticRegression()
fitModel = classifier.fit(train)


# ## Test (Evaluate)

# In[8]:


from pyspark.ml.evaluation import *

predictionAndLabels = fitModel.transform(test)
# predictionAndLabels = predictionAndLabels.predictions.select('label','prediction')
Bin_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="prediction"
)  # labelCol='label'
auc = Bin_evaluator.evaluate(predictionAndLabels)
print("AUC:", auc)

predictions = fitModel.transform(test)
MC_evaluator = MulticlassClassificationEvaluator(
    metricName="accuracy"
)  # redictionCol="prediction",
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print(
    "Accuracy: {0:.2f}".format(accuracy), "%"
)  #     print("Test Error = %g " % (1.0 - accuracy))
print(" ")


# ## Great!
#
# We did it! But usually you'll want to test several different algorithms and compare their performance as you try to solve machine learning problems. You generally don't just test out only one and move on. So I've created the function below to do just that. Hopefully it make it easier for you! Once you've selected the best model from here, you can fine tune.

# ## Create all encompassing Classification Training and Evaluation Function
#
# This function also us to iterativley pass through any classifier and train and evaluate it.

# In[16]:


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
            featureimportances = BestModel.featureImportances.toArray()
            print(featureimportances)

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


# In[13]:


# Run!
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions

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

train, test = final_data.randomSplit([0.7, 0.3])
features = final_data.select(["features"]).collect()
# Learn how many classes there are in order to specify evaluation type based on binary or multi and turn the df into an object
class_count = final_data.select(countDistinct("label")).collect()
classes = class_count[0][0]

# set up your results table
columns = ["Classifier", "Result"]
vals = [("Place Holder", "N/A")]
results = spark.createDataFrame(vals, columns)

for classifier in classifiers:
    new_result = ClassTrainEval(classifier, features, classes, train, test)
    results = results.union(new_result)
results = results.where("Classifier!='Place Holder'")
results.show(100, False)


# ### Classification Diagnostics
#
# You can also generate some more detailed diagnostics on individual classifiers using this function too if you want. The output is pretty extensive, so I wouldn't do more than one at a time if I were you.

# In[14]:


from pyspark.ml.classification import *
from pyspark.ml.evaluation import *


def ClassDiag(classifier):

    # Fit our model
    C = classifier
    fitModel = C.fit(train)

    # Load the Summary
    trainingSummary = fitModel.summary

    # General Describe
    trainingSummary.predictions.describe().show()

    # View Predictions
    pred_and_labels = fitModel.evaluate(test)
    pred_and_labels.predictions.show()

    # Print the coefficients and intercept for multinomial logistic regression
    print("Coefficients: \n" + str(fitModel.coefficientMatrix))
    print(" ")
    print("Intercept: " + str(fitModel.interceptVector))
    print(" ")

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print(" ")
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # for multiclass, we can inspect metrics on a per-label basis
    print(" ")
    print("False positive rate by label:")
    for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
        print("label %d: %s" % (i, rate))

    print(" ")
    print("True positive rate by label:")
    for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
        print("label %d: %s" % (i, rate))

    print(" ")
    print("Precision by label:")
    for i, prec in enumerate(trainingSummary.precisionByLabel):
        print("label %d: %s" % (i, prec))

    print(" ")
    print("Recall by label:")
    for i, rec in enumerate(trainingSummary.recallByLabel):
        print("label %d: %s" % (i, rec))

    print(" ")
    print("F-measure by label:")
    for i, f in enumerate(trainingSummary.fMeasureByLabel()):
        print("label %d: %s" % (i, f))

    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print(" ")
    print(
        "Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
        % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall)
    )


# In[15]:


# classifier = LogisticRegression()
ClassDiag(LogisticRegression())


# In[ ]:

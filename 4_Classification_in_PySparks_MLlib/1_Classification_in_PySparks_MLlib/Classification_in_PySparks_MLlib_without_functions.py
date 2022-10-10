#!/usr/bin/env python
# coding: utf-8

# # Classification in PySpark's MLlib
#
# PySpark offers a good variety of algorithms that can be applied to classification machine learning problems. However, because PySpark operates on distributed dataframes, we cannot use popular Python libraries like scikit learn for our machine learning applications. Which means we need to use PySpark's MLlib packages for these tasks. Luckily, MLlib offers a pretty good variety of algorithms! In this notebook we will go over how to prep our data and train and test the classification algorithms PySpark offers.
#
# ## Defining Classification
#
# As we went over in the concept review lecture, classification is a supervised machine learning task where we want to automatically categorize our data into some pre-defined categorization method. Examples of classification might include sorting objects like flowers into various species or automatically labeling images into groups like cat, dog, fish, etc. To be able to do this though, we need to have training data and a pre-defined dependent variable which is the column in your dataset that defines the categories you want to predict.
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
spark = SparkSession.builder.appName("Classification").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark
# Click the hyperlinked "Spark UI" link to view details about your Spark session


# In[2]:


# Read in functions we will need
from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

# ## Let's read our dataset in for this notebook
#
# ### Data Set Name: Autistic Spectrum Disorder Screening Data for Adult
# Autistic Spectrum Disorder (ASD) is a neurodevelopment condition associated with significant healthcare costs, and early diagnosis can significantly reduce these. Unfortunately, waiting times for an ASD diagnosis are lengthy and procedures are not cost effective. The economic impact of autism and the increase in the number of ASD cases across the world reveals an urgent need for the development of easily implemented and effective screening methods. Therefore, a time-efficient and accessible ASD screening is imminent to help health professionals and inform individuals whether they should pursue formal clinical diagnosis. The rapid growth in the number of ASD cases worldwide necessitates datasets related to behaviour traits. However, such datasets are rare making it difficult to perform thorough analyses to improve the efficiency, sensitivity, specificity and predictive accuracy of the ASD screening process. Presently, very limited autism datasets associated with clinical or screening are available and most of them are genetic in nature. Hence, we propose a new dataset related to autism screening of adults that contained 20 features to be utilised for further analysis especially in determining influential autistic traits and improving the classification of ASD cases. In this dataset, we record ten behavioural features (AQ-10-Adult) plus ten individuals characteristics that have proved to be effective in detecting the ASD cases from controls in behaviour science.
#
# ### Source:
# https://www.kaggle.com/faizunnabi/autism-screening

# In[3]:


path = "Datasets/"
df = spark.read.csv(
    path + "Toddler Autism dataset July 2018.csv", inferSchema=True, header=True
)


# ### Check out the dataset

# In[4]:


df.limit(6).toPandas()


# In[5]:


df.printSchema()


# ### How many classes do we have?
#
# It's important to check for class imbalance in your dependent variable for classification tasks. If there are extremley under or over represented classes, the accuracy of your model predictions might suffer as a result of your model essentially being biased.
#
# If you see class imbalance, one common way to correct this would be boot strapping or resampling your dataframe.

# In[6]:


df.groupBy("Class/ASD Traits ").count().show(100)


# ## Format Data
#
# MLlib requires all input columns of your dataframe to be vectorized. You will see that we rename our dependent var to label as that is what is expected for all MLlib applications. If rename once here, we never have to do it again!
#
# For more methods on transformations visit: https://spark.apache.org/docs/latest/ml-features

# In[7]:


# Declare values you will need

# col_list = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","Age_Mons","Qchat-10-Score","Sex","Ethnicity","Jaundice","Family_mem_with_ASD","Who completed the test"]
# input_columns = col_list

input_columns = df.columns  # Collect the column names as a list
input_columns = input_columns[1:-1]  # keep only relevant columns: from column 1 to

dependent_var = "Class/ASD Traits "


# In[8]:


# change label (class variable) to string type to prep for reindexing
# Pyspark is expecting a zero indexed integer for the label column.
# Just in case our data is not in that format... we will treat it by using the StringIndexer built in method
renamed = df.withColumn(
    "label_str", df[dependent_var].cast(StringType())
)  # Rename and change to string type
indexer = StringIndexer(
    inputCol="label_str", outputCol="label"
)  # Pyspark is expecting the this naming convention
indexed = indexer.fit(renamed).transform(renamed)


# In[9]:


# Convert all string type data in the input column list to numeric
# Otherwise the Algorithm will not be able to process it

# Also we will use these lists later on
numeric_inputs = []
string_inputs = []
for column in input_columns:
    # First identify the string vars in your input column list
    if str(indexed.schema[column].dataType) == "StringType":
        # Set up your String Indexer function
        indexer = StringIndexer(inputCol=column, outputCol=column + "_num")
        # Then call on the indexer you created here
        indexed = indexer.fit(indexed).transform(indexed)
        # Rename the column to a new name so you can disinguish it from the original
        new_col_name = column + "_num"
        # Add the new column name to the string inputs list
        string_inputs.append(new_col_name)
    else:
        # If no change was needed, take no action
        # And add the numeric var to the num list
        numeric_inputs.append(column)


# #### Treating for skewness and outliers
#
# Recall from the Model Selection and Tuning lecture that Skewness measures how much a distribution of values deviates from symmetry around the mean. A value of zero means the distribution is symmetric, while a positive skewness indicates a greater number of smaller values, and a negative value indicates a greater number of larger values.
#
# As a general rule of thumb:
#
#  - If skewness is **less than -1 or greater than 1**, the distribution is highly skewed.
#  - If skewness is **between -1 and -0.5 or between 0.5 and 1**, the distribution is moderately skewed.
#  - If skewness is **between -0.5 and 0.5**, the distribution is approximately symmetric.
#
# A common recommendation for treating skewness is either a log transformation for positive skewed data or an exponential transformation for negatively skewed data.
#
#
# **Outliers** <br>
# One common way to correct outliers is by flooring and capping which means editing any value that is above or below a certain threshold (99th percentile or 1st percentile) back to the highest/lowest value in that percentile. For example, if the 99th percentile is 96 and there is a value of 1,000, you would change that value to 96.

# In[10]:


# Treat for skewness
# Flooring and capping
# Plus if right skew take the log +1
# if left skew do exp transformation
# This is best practice

# create empty dictionary d
d = {}
# Create a dictionary of quantiles from your numeric cols
# I'm doing the top and bottom 1% but you can adjust if needed
for col in numeric_inputs:
    d[col] = indexed.approxQuantile(
        col, [0.01, 0.99], 0.25
    )  # if you want to make it go faster increase the last number

# Now check for skewness for all numeric cols
for col in numeric_inputs:
    skew = indexed.agg(skewness(indexed[col])).collect()  # check for skewness
    skew = skew[0][0]
    # If skewness is found,
    # This function will make the appropriate corrections
    if skew > 1:  # If right skew, floor, cap and log(x+1)
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
            col + " has been treated for positive (right) skewness. (skew =)", skew, ")"
        )
    elif skew < -1:  # If left skew floor, cap and exp(x)
        indexed = indexed.withColumn(
            col,
            exp(
                when(df[col] < d[col][0], d[col][0])
                .when(indexed[col] > d[col][1], d[col][1])
                .otherwise(indexed[col])
            ).alias(col),
        )
        print(
            col + " has been treated for negative (left) skewness. (skew =", skew, ")"
        )


# In[11]:


# Now check for negative values in the dataframe.
# Produce a warning if there are negative values in the dataframe that Naive Bayes cannot be used.
# Note: we only need to check the numeric input values since anything that is indexed won't have negative values

# Calculate the mins for all columns in the df
minimums = df.select([min(c).alias(c) for c in df.columns if c in numeric_inputs])
# Create an array for all mins and select only the input cols
min_array = minimums.select(array(numeric_inputs).alias("mins"))
# Collect golobal min as Python object
df_minimum = min_array.select(array_min(min_array.mins)).collect()
# Slice to get the number itself
df_minimum = df_minimum[0][0]

# If there are ANY negative vals found in the df, print a warning message
if df_minimum < 0:
    print(
        "WARNING: The Naive Bayes Classifier will not be able to process your dataframe as it contains negative values"
    )
else:
    print("No negative values were found in your dataframe.")


# In[12]:


# Before we correct for negative values that may have been found above,
# We need to vectorize our df
# becauase the function that we use to make that correction requires a vector.
# Now create your final features list
features_list = numeric_inputs + string_inputs
# Create your vector assembler object
assembler = VectorAssembler(inputCols=features_list, outputCol="features")
# And call on the vector assembler to transform your dataframe
output = assembler.transform(indexed).select("features", "label")


# In[13]:


# Create the mix max scaler object
# This is what will correct for negative values
# I like to use a high range like 1,000
#     because I only see one decimal place in the final_data.show() call
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures", min=0, max=1000)
print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))

# Compute summary statistics and generate MinMaxScalerModel
scalerModel = scaler.fit(output)

# rescale each feature to range [min, max].
scaled_data = scalerModel.transform(output)
final_data = scaled_data.select("label", "scaledFeatures")
# Rename to default value
final_data = final_data.withColumnRenamed("scaledFeatures", "features")
final_data.show()


# ### Split into Test and Training datasets
#
# Now we can split into test and trainging datasets using whatever random split method we want. I will use 70/30 split but you can use your own.

# In[14]:


train, test = final_data.randomSplit([0.7, 0.3])


# ## Train!
#
# Now that we have our data cleaned and vectorized we are ready to feed it into our training algorithms! As we went over in the Intro to Machine Learning lecture, the building blocks of a supervised ML application consist of some data for the model to "learn" from. Once there is data made available, then the person building the model must decide what the apprpriate dependent and independent variables are. Then they decide which algorithms to test, and compare the performance results of each model to each other before deciding which one to select.
#
# This process usually requires several trails until a decision is reached and diligent note-taking. This first notebook will go over each of the alogorithms that PySparks offers for classification and then in a later lecture, we will go over other methods for organizing your trail and error record keeping. But for now, let's try to get a handle on the basics!
#
# This portion of the course is going to be a code review instead of a code along activity as much of the code is going to be repeatable. I thought you guys might get bored so this method will help stream line things along a bit.

# In[15]:


# First - Read in dependencies
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import *

# In[29]:


# Set up our evaluation objects
Bin_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="prediction"
)  # labelCol='label'
# Bin_evaluator = BinaryClassificationEvaluator() #labelCol='label'
MC_evaluator = MulticlassClassificationEvaluator(
    metricName="accuracy"
)  # redictionCol="prediction",


# In[21]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC", evaluator.evaluate(predictions))


# ### Logistic Regression
#
# **Review**
# The Logistic Regression Algorithm, also known as "Logit", is used to estimate (guess) the probability (a number between 0 and 1) of an event occurring having been given some previous data to “learn” from. It works with either binary or multinomial (more than 2 categories) data and uses logistic function (ie. log) to find a model that fits with the data points.
#
# **Example**
# You may want to predict the likelihood of a student passing or failing an exam based on a set of biographical factors. The model you create will provide a probability (i.e a number between 0 and 1) that you can use to determine the likelihood of each student passing.
#
# PySpark Documentation Link: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression

# In[27]:


predictions = fitModel.transform(test)
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC", evaluator.evaluate(predictions))


# In[30]:


# This is the most simplistic approach which does not use cross validation
# Let's go ahead and train a Logistic Regression Algorithm
classifier = LogisticRegression()
fitModel = classifier.fit(train)

# Evaluation method for binary classification problem
predictionAndLabels = fitModel.transform(test)
auc = Bin_evaluator.evaluate(predictionAndLabels)
print("AUC:", auc)

# Evaluation for a multiclass classification problem
predictions = fitModel.transform(test)
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print(
    "Accuracy: {0:.2f}".format(accuracy), "%"
)  #     print("Test Error = %g " % (1.0 - accuracy))
print(" ")


# #### Add in cross validation
#
# Spark also has a built-in funciton called the CrossValidator to conduct cross validation which begins by splitting the training dataset into a set of "folds" which are used as separate training and test datasets. For example, with k=5 folds, CrossValidator will generate 5 different (training, test) dataset pairs, each of which uses 4/5 of the data for training and 1/5 for testing. To evaluate a particular Parameter (specified in the paramgrid), CrossValidator computes the average evaluation metric for the 5 Models produced by fitting the Estimator on the 5 different (training, test) dataset pairs and tells you which model performed the best once it is finished.
#
# After identifying the best ParamMap (more details can be found in the documentation link above), CrossValidator finally re-fits the Estimator using the best ParamMap and the entire dataset.
#
# **MaxIter:** <br>
# The maximum number of iterations to use. There is no clear formula for setting the optimum iteration number, but you can figure out this issue by an iterative process by initializing the iteration number by a small number like 100 and then increase it linearly. This process will be repeated until the MSE of the test does not decrease and even may increase. The below link describes well:
# https://www.quora.com/What-will-happen-if-I-train-my-neural-networks-with-too-much-iteration

# In[31]:


# First tell Spark which classifier you want to use
classifier = LogisticRegression()

# Then Set up your parameter grid for the cross validator to conduct hyperparameter tuning
paramGrid = ParamGridBuilder().addGrid(classifier.maxIter, [10, 15, 20]).build()

# Then set up the Cross Validator which requires all of the following parameters:
crossval = CrossValidator(
    estimator=classifier,
    estimatorParamMaps=paramGrid,
    evaluator=MC_evaluator,
    numFolds=2,
)  # 3 + is best practice

# Then fit your model
fitModel = crossval.fit(train)

# Collect the best model and
# print the coefficient matrix
# These values should be compared relative to eachother
# And intercepts can be prepared to other models
BestModel = fitModel.bestModel
print("Intercept: " + str(BestModel.interceptVector))
print("Coefficients: \n" + str(BestModel.coefficientMatrix))

# You can extract the best model from this run like this if you want
LR_BestModel = BestModel

# Next you need to generate predictions on the test dataset
# fitModel automatically uses the best model
# so we don't need to use BestModel here
predictions = fitModel.transform(test)

# Now print the accuracy rate of the model or AUC for a binary classifier
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print(accuracy)


# In[32]:


# zip input_columns qith feature importance scores and create df

# First convert featureimportance scores from numpy array to list
coeff_array = BestModel.coefficientMatrix.toArray()
coeff_scores = []
for x in coeff_array[0]:
    coeff_scores.append(float(x))
# Then zip with input_columns list and create a df

# data_schema = [StructField("feature", StringType(), True),StructField("coeff", DecimalType(), True)]
# final_struc = StructType(fields=data_schema)
# result = spark.createDataFrame(zip(input_columns,coeff_scores), schema=final_struc)

result = spark.createDataFrame(
    zip(input_columns, coeff_scores), schema=["feature", "coeff"]
)
result.show(100)


# 100% accuracy! Pretty good eh? Well this is probably just because we did not have much test data. You will almost never see this happen. If you know more about the subject matter of the data, you can compare the coefficients you see here. If you are not familiar with this concept, here is a good article to help you understand: https://www.displayr.com/how-to-interpret-logistic-regression-coefficients/

# ### Classification Diagnostics
#
# You can also generate some more detailed diagnostics too if you want.
#
# To learn more about it please visit: https://towardsdatascience.com/demystifying-confusion-matrix-confusion-9e82201592fd. Please note that this article provides coding examples for scikit learn (not PySpark) so please do not let that confuse you. However I found the explaination and approach to be very helpful.
#
#
# *Note: This output is ONLY available for Logistic Regression*

# In[34]:


# Load the Summary
trainingSummary = LR_BestModel.summary

# General Describe
trainingSummary.predictions.describe().show()

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print(" ")
print("objectiveHistory: (scaled loss + regularization) at each iteration")
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

# Generate confusion matrix and print (includes accuracy)
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


# ### One vs. Rest
#
# **Recap from lecture**
# The One-vs-Rest classifier is a type of multiclass classifier that involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. So each class is viewed as it compares to rest of the classes as a whole, as opposed to each one individually.
#
# **regParam**: <br>
# The purpose of the regularizer is to encourage simple models and avoid overfitting. To learn more about this concept, here is an interesting article: https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a
#
#
# PySpark Documentation link: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.OneVsRest

# In[35]:


# instantiate the base classifier.
lr = LogisticRegression()
# instantiate the One Vs Rest Classifier.
classifier = OneVsRest(classifier=lr)

# Add parameters of your choice here:
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
# Cross Validator requires the following parameters:
crossval = CrossValidator(
    estimator=classifier,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassClassificationEvaluator(),
    numFolds=2,
)  # 3 is best practice

# Run cross-validation, and choose the best set of parameters.
fitModel = crossval.fit(train)

# Print the Coefficients
# First we need to extract the best model from fit model

# Get Best Model
BestModel = fitModel.bestModel
# Extract list of binary models
models = BestModel.models
for model in models:
    print(
        "\033[1m" + "Intercept: " + "\033[0m",
        model.intercept,
        "\033[1m" + "\nCoefficients:" + "\033[0m",
        model.coefficients,
    )

# Now generate predictions on test dataset
predictions = fitModel.transform(test)
# And calculate the accuracy score
accuracy = (MC_evaluator.evaluate(predictions)) * 100
# And print
print(accuracy)


# ## Multilayer Perceptron Classifier
#
# *Neural Network* <br>
#
# **Recap from the lecture** <br>
# A multilayer perceptron (MLP) is a class of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.
#
# #### Common Hyper Parameters:
#
# **MaxIter:** <br>
# The maximum number of iterations to use. There is no clear formula for setting the optimum iteration number, but you can figure out this issue by an iterative process by initializing the iteration number by a small number like 100 and then increase it linearly. This process will be repeated until the MSE of the test does not decrease and even may increase. The below link describes well:
# https://www.quora.com/What-will-happen-if-I-train-my-neural-networks-with-too-much-iteration
#
# **Layers:** <br>
# Spark requires that the input layer equals the number of features in the dataset, the hidden layer might be one or two more than that (flexible), and the output layer has to be equal to the number of classes. Here's a great article to learn more about how to play around with the hidden layers: https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e
#
# **Block size:** <br>
# Block size for stacking input data in matrices to speed up the computation. Data is stacked within partitions. If block size is more than remaining data in a partition then it is adjusted to the size of this data. Recommended size is between 10 and 1000. Default: 128
#
# **Seed:** <br>
# A random seed. Set this value if you need your results to be reproducible across repeated calls (highly recommdended).
#
# **Weights**: *printed for us below along with accuracy rate* <br>
# Each hidden neuron added will increase the number of weights, thus it is recommended to use the least number of hidden neurons that accomplish the task. Using more hidden neurons than required will add more complexity.
#
# **PySpark Documentation link:** <br>
# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.MultilayerPerceptronClassifier

# In[36]:


# Count how many features you have
features = final_data.select(["features"]).collect()
features_count = len(features[0][0])
# Count how many classes you have
class_count = final_data.select(countDistinct("label")).collect()
classes = class_count[0][0]

# Then use this number to specify the layers
# The first number in this list is the input layer which has to be equal to the number of features in your vector
# The second number is the first hidden layer
# The third number is the second hidden layer
# The fourth number is the output layer which has to be equal to your class size
layers = [features_count, features_count + 1, features_count, classes]
# Instaniate the classifier
classifier = MultilayerPerceptronClassifier(
    maxIter=100, layers=layers, blockSize=128, seed=1234
)

# Fit the model
fitModel = classifier.fit(train)

# Print the model Weights
print("\033[1m" + "Model Weights: " + "\033[0m", fitModel.weights.size)

# Generate predictions on test dataframe
predictions = fitModel.transform(test)
# Calculate accuracy score
accuracy = (MC_evaluator.evaluate(predictions)) * 100
# Print accuracy score
print("Accuracy: ", accuracy)


# ### Naive Bayes
#
# **Recap from the lecture:**
# The Naive Bayes Classifier is a collection of classification algorithms based on Bayes Theorem. It is not a single algorithm but a family of algorithms that all share a common principle, that every feature being classified is independent of the value of any other feature.
#
# So for example, a fruit may be considered to be an apple if it is red, round, and about 3″ in diameter. A Naive Bayes classifier considers each of these “features” (red, round, 3” in diameter) to contribute independently to the probability that the fruit is an apple, regardless of any correlations between features. Features, however, aren’t always independent which is often seen as a shortcoming of the Naive Bayes algorithm and this is why it’s labeled “naive”.
#
# **Assumptions:**
#  - Independence between every pair of features
#  - Feature values are nonnegative (which is why we checked earlier)
#
# **Hyper Parameters:**
#
#  - **smoothing** = It is problematic when a frequency-based probability is zero, because it will wipe out all the information in the other probabilities, and we need to find a solution for this. A solution would be Laplace smoothing , which is a technique for smoothing categorical data. In PySpark, this number needs to be be >= 0, default is 1.0'. Also here is a great article that defines smoothing in more detail: https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf
#  - **thresholds** = Thresholds in multi-class classification to adjust the probability of predicting each class. Array must have length equal to the number of classes, with values > 0, excepting that at most one value may be 0. The class with largest value p/t is predicted, where p is the original probability of that class and t is the class's threshold. The default value is none.
#  - **weightCol** = If you have a weight column you would enter the name of the column here. If this is not set or empty, we treat all instance weights as 1.0. To learn more about the theory behind this, here is a good paper: http://pami.uwaterloo.ca/~khoury/ece457f07/Zhang2004.pdf

# In[37]:


# Add parameters of your choice here:
classifier = NaiveBayes()
paramGrid = (
    ParamGridBuilder().addGrid(classifier.smoothing, [0.0, 0.2, 0.4, 0.6]).build()
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

predictions = fitModel.transform(test)
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print("Accuracy: ", accuracy)


# ## Linear Support Vector Machine
#
# **Recap from lecture:**
# Linear SVMs are based on the idea of finding a hyperplane that best divides a dataset into two classes, which is why you can only use it for binary classification. Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter the position of the dividing hyperplane. Because of this, they can be considered the critical elements of a data set. Intuitively, the further from the hyperplane our data points lie, the more confident we are that they have been correctly classified. We therefore want our data points to be as far away from the hyperplane as possible, while still being on the correct side of it. So when new testing data is added, whatever side of the hyperplane it lands will decide the class that we assign to it.
#
# **Interpretting the coefficients:**
#
# Each coefficients direction gives us the predicted class, so if you take the dot product of any point with the vector, you can tell on which side it is: if the dot product is positive, it belongs to the positive class, if it is negative it belongs to the negative class.
#
# You can even learn something about the importance of each feature. Let's say the svm would find only one feature useful for separating the data, then the hyperplane would be orthogonal to that axis. So, you could say that the absolute size of the coefficient relative to the other ones gives an indication of how important the feature was for the separation.
#
# **Hyper Parameters:** <br>
#
# **MaxIter:** <br>
# The maximum number of iterations to use. There is no clear formula for setting the optimum iteration number, but you can figure out this issue by an iterative process by initializing the iteration number by a small number like 100 and then increase it linearly. This process will be repeated until the MSE of the test does not decrease and even may increase. The below link describes well:
# https://www.quora.com/What-will-happen-if-I-train-my-neural-networks-with-too-much-iteration
#
# **regParam**: <br>
# The purpose of the regularizer is to encourage simple models and avoid overfitting. To learn more about this concept, here is an interesting article: https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a
#
# **PySpark Documentation link:** <br> https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LinearSVC

# In[38]:


# Count how many classes you have and produce an error if it's more than 2.
class_count = final_data.select(countDistinct("label")).collect()
classes = class_count[0][0]
if classes > 2:
    print(
        "LinearSVC cannot be used because PySpark currently only accepts binary classification data for this algorithm"
    )

# Add parameters of your choice here:
classifier = LinearSVC()
paramGrid = (
    ParamGridBuilder()
    .addGrid(classifier.maxIter, [10, 15])
    .addGrid(classifier.regParam, [0.1, 0.01])
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

BestModel = fitModel.bestModel

print("Intercept: \n" + str(BestModel.intercept))
print("\033[1m" + " Coefficients" + "\033[0m")
print("You should compares these relative to eachother")
print("Coefficients: \n" + str(BestModel.coefficients))

# Automatically gets the best model
predictions = fitModel.transform(test)
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print("Accuracy: ", accuracy)


# ## Decision Tree
#
# **Recall from the lecture:**
# Decision Trees classifiers  are a supervised learning method is used to classify a variable by learning from historical data that the model uses to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model.
#
# Decision tree builds classification or regression models in the form of a tree structure. It breaks down a data set into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node has two or more branches. Leaf node represents a classification or decision. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.
#
# ### Common Hyper Parameters
#
#  - **maxBins** = Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature.
#      - **Continuous features:** For small datasets in single-machine implementations, the split candidates for each continuous feature are typically the unique values for the feature. Some implementations sort the feature values and then use the ordered unique values as split candidates for faster tree calculations.
#          Sorting feature values is expensive for large distributed datasets. This implementation computes an approximate set of split candidates by performing a quantile calculation over a sampled fraction of the data. The ordered splits create “bins” and the maximum number of such bins can be specified using the maxBins parameter.
#          Note that the number of bins cannot be greater than the number of instances N (a rare scenario since the default maxBins value is 32). The tree algorithm automatically reduces the number of bins if the condition is not satisfied.
#
#      - **Categorical features:** For a categorical feature with M possible values (categories), one could come up with 2 exp(M−1) −1 split candidates. For binary (0/1) classification and regression, we can reduce the number of split candidates to M−1 by ordering the categorical feature values by the average label. For example, for a binary classification problem with one categorical feature with three categories A, B and C whose corresponding proportions of label 1 are 0.2, 0.6 and 0.4, the categorical features are ordered as A, C, B. The two split candidates are A | C, B and A , C | B where | denotes the split.
#          In multiclass classification, all 2 exp(M−1) −1 possible splits are used whenever possible. When 2 exp(M−1) −1 is greater than the maxBins parameter, we use a (heuristic) method similar to the method used for binary classification and regression. The M categorical feature values are ordered by impurity, and the resulting M−1 split candidates are considered.
#
#  - **maxDepth** = The max_depth parameter specifies the maximum depth of each tree. The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.
#
# ### Feature Importance Scores
# Scores add up to 1 accross all varaibles so the lowest score is the least imporant variable.
#
#
# ### Extra Reading
# **How to tune a decision tree** <br>
# https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
#
# **PySpark Documentation link:** <br> https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassifier

# In[39]:


# Add parameters of your choice here:
classifier = DecisionTreeClassifier()
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

# Collect and print feature importances
BestModel = fitModel.bestModel
featureImportances = BestModel.featureImportances.toArray()
print("Feature Importances: ", featureImportances)

predictions = fitModel.transform(test)
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print("Accuracy: ", accuracy)


# In[40]:


# zip input_columns qith feature importance scores and create df

# First convert featureimportance scores from numpy array to list
imp_scores = []
for x in featureImportances:
    imp_scores.append(int(x))

# Then zip with input_columns list and create a df
result = spark.createDataFrame(
    zip(input_columns, imp_scores), schema=["feature", "score"]
)
print(result.orderBy(result["score"].desc()).show(truncate=False))


# ## Random Forest
#
# **Recal from the lecture** <br>
# Suppose you have a training set with 6 classes, random forest may create three decision trees taking input of each subset. Finally, it predicts based on the majority of votes from each of the decision trees made. This works well because a single decision tree may be prone to noise, but aggregate of many decision trees reduce the effect of noise giving more accurate results. The subsets in different decision trees created may overlap.
#
#
# ### Common Hyper Parameters
#
#  - **maxBins** = Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature.
#      - **Continuous features:** For small datasets in single-machine implementations, the split candidates for each continuous feature are typically the unique values for the feature. Some implementations sort the feature values and then use the ordered unique values as split candidates for faster tree calculations.
#          Sorting feature values is expensive for large distributed datasets. This implementation computes an approximate set of split candidates by performing a quantile calculation over a sampled fraction of the data. The ordered splits create “bins” and the maximum number of such bins can be specified using the maxBins parameter.
#          Note that the number of bins cannot be greater than the number of instances N (a rare scenario since the default maxBins value is 32). The tree algorithm automatically reduces the number of bins if the condition is not satisfied.
#
#      - **Categorical features:** For a categorical feature with M possible values (categories), one could come up with 2 exp(M−1) −1 split candidates. For binary (0/1) classification and regression, we can reduce the number of split candidates to M−1 by ordering the categorical feature values by the average label. For example, for a binary classification problem with one categorical feature with three categories A, B and C whose corresponding proportions of label 1 are 0.2, 0.6 and 0.4, the categorical features are ordered as A, C, B. The two split candidates are A | C, B and A , C | B where | denotes the split.
#          In multiclass classification, all 2 exp(M−1) −1 possible splits are used whenever possible. When 2 exp(M−1) −1 is greater than the maxBins parameter, we use a (heuristic) method similar to the method used for binary classification and regression. The M categorical feature values are ordered by impurity, and the resulting M−1 split candidates are considered.
#
#  - **maxDepth** = The maxDepth parameter specifies the maximum depth of each tree. The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.
#
# ### Feature Importance Scores
# Scores add up to 1 accross all varaibles so the lowest score is the least imporant variable.
#
# PySpark Documentation link: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier

# In[41]:


# Add parameters of your choice here:
classifier = RandomForestClassifier()
paramGrid = (
    ParamGridBuilder().addGrid(classifier.maxDepth, [2, 5, 10])
    #                                .addGrid(classifier.maxBins, [5, 10, 20])
    #                                .addGrid(classifier.numTrees, [5, 20, 50])
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

# Retrieve best model from cross val
BestModel = fitModel.bestModel
featureImportances = BestModel.featureImportances.toArray()
print("Feature Importances: ", featureImportances)

predictions = fitModel.transform(test)

accuracy = (MC_evaluator.evaluate(predictions)) * 100
print(" ")
print("Accuracy: ", accuracy)


# ## Gradient Boost Tree Classifier
#
# **Recall from the lecture**
# With gradient boosting, it’s more of a hierarchical approach. It combines the weak learners (binary splits) to strong prediction rules that allow a flexble partition of the feature space. The objective here, as is of any supervised learning algorithm, is to define a loss function and minimize it.
#
# ### Common Hyper Parameters
#
#  - **maxBins** = Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature.
#      - **Continuous features:** For small datasets in single-machine implementations, the split candidates for each continuous feature are typically the unique values for the feature. Some implementations sort the feature values and then use the ordered unique values as split candidates for faster tree calculations.
#          Sorting feature values is expensive for large distributed datasets. This implementation computes an approximate set of split candidates by performing a quantile calculation over a sampled fraction of the data. The ordered splits create “bins” and the maximum number of such bins can be specified using the maxBins parameter.
#          Note that the number of bins cannot be greater than the number of instances N (a rare scenario since the default maxBins value is 32). The tree algorithm automatically reduces the number of bins if the condition is not satisfied.
#
#      - **Categorical features:** For a categorical feature with M possible values (categories), one could come up with 2 exp(M−1) −1 split candidates. For binary (0/1) classification and regression, we can reduce the number of split candidates to M−1 by ordering the categorical feature values by the average label. For example, for a binary classification problem with one categorical feature with three categories A, B and C whose corresponding proportions of label 1 are 0.2, 0.6 and 0.4, the categorical features are ordered as A, C, B. The two split candidates are A | C, B and A , C | B where | denotes the split.
#          In multiclass classification, all 2 exp(M−1) −1 possible splits are used whenever possible. When 2 exp(M−1) −1 is greater than the maxBins parameter, we use a (heuristic) method similar to the method used for binary classification and regression. The M categorical feature values are ordered by impurity, and the resulting M−1 split candidates are considered.
#
#  - **maxDepth** = The maxDepth parameter specifies the maximum depth of each tree. The default value for max_depth is None, which means that each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class.
#
# ### Feature Importance Scores
# Scores add up to 1 accross all varaibles so the lowest score is the least imporant variable.
#
# PySpark Documentation link: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.GBTClassifier

# In[42]:


class_count = final_data.select(countDistinct("label")).collect()
classes = class_count[0][0]
if classes > 2:
    print(
        "GBTClassifier cannot be used because PySpark currently only accepts binary classification data for this algorithm"
    )

# Add parameters of your choice here:
classifier = GBTClassifier()

paramGrid = (
    ParamGridBuilder()  #                              .addGrid(classifier.maxDepth, [2, 5, 10, 20, 30]) \
    #                              .addGrid(classifier.maxBins, [10, 20, 40, 80, 100]) \
    .addGrid(classifier.maxIter, [10, 15, 50, 100]).build()
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

BestModel = fitModel.bestModel
featureImportances = BestModel.featureImportances.toArray()
print("Feature Importances: ", featureImportances)

predictions = fitModel.transform(test)
accuracy = (MC_evaluator.evaluate(predictions)) * 100
print(" ")
print("Accuracy: ", accuracy)


# ## That's it!
#
# Great job guys!
#
# ### Next up
#
# We will learn how to add functions to this script to make it a bit easier to use.

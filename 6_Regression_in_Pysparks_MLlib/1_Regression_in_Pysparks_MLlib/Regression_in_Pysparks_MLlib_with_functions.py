#!/usr/bin/env python
# coding: utf-8

# # Regression in PySpark's MLlib
#
# PySpark offers 7 algorithms for regression which we will review in this lecture. The content of this notebook will be very similar to what we did in the classification lectures where we will see how to prep our data first, and then go over how to train and evaluate each model individually.
#
# **Recap from the Regression lecture**<br>
# Remember that regression problems require that the **dependent variable** in your dataset be continuous (like age or height) and not categorical like (young vs old, or fat vs skinny). Regression analysis tries to find the relationship between this variable (the dependent) and each of the independent variables which can be either continous or categorical. As with any machine learning problem, the basic question of regression analysis is "what factors affect our outcome."
#
# For example, some research questions that might be solved using regression anlaysis might be:
#
# What factors effect...
#
# 1. inflation rate and how we predict it longer term?
# 2. price increases upon demand
# 3. the height or weight of a person
# 4. crop yield of vegetation like corn or apple trees
# 5. income of a person
#
#
# ## Available Algorithms
# These are the regression algorithms Spark offers:
#
# 1. Linear regression
#      - most simplistic and easy to understand
# 2. Generalized linear regression (out of scope)
# 3. Decision tree regression
#      - most basic of the tree algorithms)
# 4. Random forest regression
#      - a bit more complex than decision tree as it is an ensemble method as it combines several models in order to produce one really great predictive model
# 5. Gradient-boosted tree regression
#      - most complex of the tree algorithms as it takes a more hierarchical approach to learning making it more efficient
# 6. Survival regression (Out of scope)
# 7. Isotonic regression (Out of scope)

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Regression").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# ## Import Dataset
#
# This is a dataset containing housing pricing data for California. Each row of data represents the median statistics for a block (eg. median income, median age of house, etc.). You could this data in a number of ways, but we will use it to predict the median house value.
#
#
# ### About this dataset
#
# 1. longitude: A measure of how far west a house is; a higher value is farther west
# 2. latitude: A measure of how far north a house is; a higher value is farther north
# 3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
# 4. totalRooms: Total number of rooms within a block
# 5. totalBedrooms: Total number of bedrooms within a block
# 6. population: Total number of people residing within a block
# 7. households: Total number of households, a group of people residing within a home unit, for a block
# 8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# 9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
# 10. oceanProximity: Location of the house w.r.t ocean/sea
#
# **Source:** https://www.kaggle.com/camnugent/california-housing-prices

# In[2]:


path = "Datasets/"
df = spark.read.csv(path + "housing.csv", inferSchema=True, header=True)


# **View data**

# In[3]:


df.limit(6).toPandas()


# **And of course the schema :)**

# In[4]:


df.printSchema()


# In[5]:


# Starting
print(df.count())
print(len(df.columns))


# Let's cut this down to a smaller dataframe size to get results faster

# In[6]:


# If you want to make your code run faster, you can slice the df like this...
# # Slice rows
# df = df.limit(300)

# # Slice columns
# cols_list = df.columns[4:9]
# df = df.select(cols_list)

# # QA
# print(df.count())
# print(len(df.columns))


# **Drop any missing values**
#
# Let's go ahead and drop any missing values for the sake of simplicity for this lecture as we have already covered the alternatives in subsequent lectures.

# In[7]:


# drop missing data
df = df.na.drop()
df.count()


# ## Format Data
#
# MLlib requires all input columns of your dataframe to be vectorized. You will see that we rename our dependent var to label as that is what is expected for all MLlib applications. If we rename once here, we won't need to specify it later on!

# In[8]:


def MLRegressDFPrep(df, input_columns, dependent_var, treat_outliers=True):

    renamed = df.withColumnRenamed(dependent_var, "label")

    # Make sure dependent variable is numeric and change if it's not
    if str(renamed.schema["label"].dataType) != "IntegerType":
        renamed = renamed.withColumn("label", renamed["label"].cast(FloatType()))

    # Convert all string type data in the input column list to numeric
    # Otherwise the Algorithm will not be able to process it
    numeric_inputs = []
    string_inputs = []
    for column in input_columns:
        if str(renamed.schema[column].dataType) == "StringType":
            new_col_name = column + "_num"
            string_inputs.append(new_col_name)
        else:
            numeric_inputs.append(column)
            indexed = renamed

    if len(string_inputs) != 0:  # If the datafraem contains string types
        for column in input_columns:
            if str(renamed.schema[column].dataType) == "StringType":
                indexer = StringIndexer(inputCol=column, outputCol=column + "_num")
                indexed = indexer.fit(renamed).transform(renamed)
    else:
        indexed = renamed

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

    # Vectorize your features
    features_list = numeric_inputs + string_inputs
    assembler = VectorAssembler(inputCols=features_list, outputCol="features")
    final_data = assembler.transform(indexed).select("features", "label")

    return final_data


# And apply it

# In[9]:


from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

input_columns = ["total_bedrooms", "population", "households", "median_income"]
dependent_var = "median_house_value"

final_data = MLRegressDFPrep(df, input_columns, dependent_var)
final_data.show(5)


# ## Check for Multicollinearity
#
# Multicollinearity generally occurs when there are high correlations between two or more predictor variables (your features column in your dataframe, also called independent variables). In other words, one predictor variable can be used to predict the other. This creates redundant information, skewing the results in a regression model.
#
# An easy way to detect multicollinearity is to calculate correlation coefficients for all pairs of predictor variables. If the correlation coefficient, is exactly +1 or -1, this is called perfect multicollinearity, and one of the variables should be removed from the model if at all possible for the linear model to perform well.
#
# Desicion trees on the other hand, make no assumptions on relationships between features. It just constructs splits on single features that improves classification, based on an impurity measure like Gini or entropy. If features A, B are heavily correlated, no /little information can be gained from splitting on B after having split on A. So it would typically get ignored in favor of C.
#
# Of course a single decision tree is very vulnerable to overfitting, so one must either limit depth, prune heavily or preferly average many using an ensemble. Such problems get worse with many features and possibly also with co-variance but this problem occurs independently from multicolinearity.
#
# MLlib offers two correlation coefficient statitics: **pearson** and **spearman**.
#
# **Sources:**
#
#  - https://datascience.stackexchange.com/questions/31402/multicollinearity-in-decision-tree
#  - https://www.statisticshowto.datasciencecentral.com/multicollinearity/

# In[10]:


from pyspark.ml.stat import Correlation

pearsonCorr = Correlation.corr(final_data, "features", "pearson").collect()[0][0]
array = pearsonCorr.toArray()


# In[11]:


for item in array:
    print(item[1])


# Looks like the first and second features are highly correlated, along with the 4th and 5th and 4th and 6th. We may want to consider removing one of the variables in each correlation pair if we decide to use a logistic regression model.

# ** Split dataframe into training and evaluation (test) dataframes**

# In[12]:


train, test = final_data.randomSplit([0.7, 0.3])


# ## Train & Evaluate!

# In[ ]:


from pyspark.ml.evaluation import *
# Dependencies for this section
from pyspark.ml.regression import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ## Most Simplistic Training method
#
# Let's train a Linear Regression algorithm to start with just to show the most simlictic way to train and test a model.

# In[13]:


# Fit our model
regressor = LinearRegression()
fitModel = regressor.fit(train)


# ## Standard Evaluation/Test method for regression (rmse/r^2)
#
# We will use the Root Mean Squared Error as our evaluation metric.

# In[14]:


# Make predictions.
predictions = fitModel.transform(test)
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# ## Define a Function that iterativley runs through all Regression algorithms
#
# **Note:**
# We did not include Generalized Linear Regression here since it requires a much different implementation method and evaluation strategy than most regressions.

# In[15]:


def RegressTrainEval(regressor):
    def FindMtype(regressor):
        # Intstantiate Model
        M = regressor
        # Learn what it is
        Mtype = type(M).__name__

        return Mtype

    Mtype = FindMtype(regressor)
    #     print('\033[1m' + Mtype + ':' + '\033[0m')

    if Mtype == "LinearRegression":

        # first without cross val
        fitModel = regressor.fit(train)

        # Load the Summary
        trainingSummary = fitModel.summary

        # Print the coefficients and intercept for linear regression
        print(
            "\033[1m"
            + "Linear Regression Model Training Summary without cross validation:"
            + "\033[0m"
        )
        print(" ")
        print("Intercept: %s" % str(fitModel.intercept))
        print("")
        print("Coefficients: ")
        coeff_array = fitModel.coefficients.toArray()
        # Convert from numpy array to list
        coeff_list = []
        for x in coeff_array:
            coeff_list.append(float(x))
        result = spark.createDataFrame(
            zip(input_columns, coeff_list), schema=["feature", "coeff"]
        )
        print(result.orderBy(result["coeff"].desc()).show(truncate=False))

        # Summarize the model over the training set and print out some metrics
        print("numIterations: %d" % trainingSummary.totalIterations)
        print(
            "objectiveHistory: (scaled loss + regularization) at each iteration \n %s"
            % str(trainingSummary.objectiveHistory)
        )
        print("")

        # Print the Errors
        print("Training RMSE: %f" % trainingSummary.rootMeanSquaredError)
        print("Training r2: %f" % trainingSummary.r2)
        print("")

        # Now load the test results
        test_results = fitModel.evaluate(test)

        # And print them
        print("Test RMSE: {}".format(test_results.rootMeanSquaredError))
        print("Test r2: {}".format(test_results.r2))
        print("")

        # Now train with cross val
        paramGrid = (
            ParamGridBuilder()  #              .addGrid(regressor.maxIter, [10, 15]) \
            .addGrid(regressor.regParam, [0.1, 0.01])
            .build()
        )

        # Evaluator
        revaluator = RegressionEvaluator(metricName="rmse")

        # Cross Validator requires all of the following parameters:
        crossval = CrossValidator(
            estimator=regressor,
            estimatorParamMaps=paramGrid,
            evaluator=revaluator,
            numFolds=2,
        )  # 3 is best practice

        print(
            "\033[1m"
            + "Linear Regression Model Summary WITH cross validation:"
            + "\033[0m"
        )
        print(" ")
        # Run cross validations
        fitModel = crossval.fit(train)

        # save model
        global LR_BestModel
        LR_BestModel = fitModel.bestModel

        print("Coefficients: ")
        coeff_array = LR_BestModel.coefficients.toArray()
        # Convert from numpy array to list
        coeff_list = []
        for x in coeff_array:
            coeff_list.append(float(x))
        result = spark.createDataFrame(
            zip(input_columns, coeff_list), schema=["feature", "coeff"]
        )
        print(result.orderBy(result["coeff"].desc()).show(truncate=False))

        # Get Model Summary Statistics
        ModelSummary = fitModel.bestModel.summary

        print("Coefficient Standard Errors: ")
        coeff_ste = ModelSummary.coefficientStandardErrors
        result = spark.createDataFrame(
            zip(input_columns, coeff_ste), schema=["feature", "coeff std error"]
        )
        print(result.orderBy(result["coeff std error"].desc()).show(truncate=False))
        print(" ")
        print("P Values: ")
        # Then zip with input_columns list and create a df
        pvalues = ModelSummary.pValues
        result = spark.createDataFrame(
            zip(input_columns, pvalues), schema=["feature", "P-Value"]
        )
        print(result.orderBy(result["P-Value"].desc()).show(truncate=False))
        print(" ")

        # Use test set here so we can measure the accuracy of our model on new data
        ModelPredictions = fitModel.transform(test)

        # cvModel uses the best model found from the Cross Validation
        # Evaluate best model
        test_results = revaluator.evaluate(ModelPredictions)
        print("RMSE:", test_results)

        # Set the column names to match the external results dataframe that we will join with later:
        columns = ["Regressor", "Result"]

        # Format results and return
        rmse_str = [str(test_results)]  # make this a string and convert to a list
        Mtype = [Mtype]  # make this a string
        result = spark.createDataFrame(zip(Mtype, rmse_str), schema=columns)
        result = result.withColumn("Result", result.Result.substr(0, 5))
        return result

    else:

        # Add parameters of your choice here:
        if Mtype in ("RandomForestRegressor"):
            paramGrid = (
                ParamGridBuilder()  #                            .addGrid(regressor.maxDepth, [2, 5, 10])
                #                            .addGrid(regressor.maxBins, [5, 10, 20])
                .addGrid(regressor.numTrees, [5, 20]).build()
            )

        # Add parameters of your choice here:
        if Mtype in ("GBTRegressor"):
            paramGrid = (
                ParamGridBuilder()  #                          .addGrid(regressor.maxDepth, [2, 5, 10, 20, 30]) \
                .addGrid(regressor.maxBins, [10, 20])
                .addGrid(regressor.maxIter, [10, 15])
                .build()
            )

        # Add parameters of your choice here:
        if Mtype in ("DecisionTreeRegressor"):
            paramGrid = (
                ParamGridBuilder()  #                          .addGrid(regressor.maxDepth, [2, 5, 10, 20, 30]) \
                .addGrid(regressor.maxBins, [10, 20, 40])
                .build()
            )

        # Cross Validator requires all of the following parameters:
        crossval = CrossValidator(
            estimator=regressor,
            estimatorParamMaps=paramGrid,
            evaluator=RegressionEvaluator(metricName="rmse"),
            numFolds=2,
        )  # 3 is best practice
        # Fit Model: Run cross-validation, and choose the best set of parameters.
        fitModel = crossval.fit(train)

        # Get Best Model
        BestModel = fitModel.bestModel

        # FEATURE IMPORTANCES
        # Estimate of the importance of each feature.
        # Each feature’s importance is the average of its importance across all trees
        # in the ensemble The importance vector is normalized to sum to 1.
        print(" ")
        print("\033[1m" + Mtype, " Feature Importances" + "\033[0m")
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

        # Create Global Variables for feature importances and models
        if Mtype in ("DecisionTreeRegressor"):
            global DT_featureImportances
            DT_featureImportances = BestModel.featureImportances.toArray()
            global DT_BestModel
            DT_BestModel = fitModel.bestModel
        if Mtype in ("GBTRegressor"):
            global GBT_featureImportances
            GBT_featureImportances = BestModel.featureImportances.toArray()
            global GBT_BestModel
            GBT_BestModel = fitModel.bestModel
        if Mtype in ("RandomForestRegressor"):
            global RF_featureImportances
            RF_featureImportances = BestModel.featureImportances.toArray()
            global RF_BestModel
            RF_BestModel = fitModel.bestModel

        # Set the column names to match the external results dataframe that we will join with later:
        columns = ["Regressor", "Result"]

        # Make predictions.
        predictions = fitModel.transform(test)
        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        rmse_str = [str(rmse)]  # make this a string and convert to a list
        Mtype = [Mtype]  # make this a string
        result = spark.createDataFrame(zip(Mtype, rmse_str), schema=columns)
        # Clean up the Result column and output
        result = result.withColumn("Result", result.Result.substr(0, 5))
        return result


# In[16]:


# Run!
regressors = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GBTRegressor(),
]

# set up your results table
columns = ["Regressor", "Result"]
vals = [("Place Holder", "N/A")]
results = spark.createDataFrame(vals, columns)

for regressor in regressors:
    new_result = RegressTrainEval(regressor)
    results = results.union(new_result)
results = results.where("Regressor!='Place Holder'")
results.show(100, False)


# ### Take aways
#
# Looks like median income was the most influencial feature in the dataset which we can see by looking at the coefficients and feature important scores, but all features had significance accross the board in the linear model.
#
# The Gradient Boosted regressor had the lowest root mean squared error (rmse) which would make it the best performing model.
#
# Now let's see how our predictions were

# In[26]:


test = final_data.limit(10)
predictions = GBT_BestModel.transform(test)
predictions = predictions.withColumn(
    "difference", predictions.prediction - predictions.label
).withColumn(
    "diff perct",
    ((predictions.prediction - predictions.label) / predictions.label) * 100,
)
print(predictions.show())
print(predictions.describe(["diff perct"]).show())


# As you can see from the output above how much difference there was between the actual house price (label) and the prediction one as both a raw dollar value and the percentage difference. The summary we printed out shows that our model, on average was off by about -9 % which means that the model under valued houses since it's a negative value. These are really great statistics to be able to deliver to a client who does have any knowledge about machine learning and wants to know how useful or "good" your model is.
#
# If you want to save your model you can simply...

# In[39]:


# this path will create the folder if it does not exist
# I also like to create a unique identifier using a timestamp so i know when the model was created
from datetime import datetime

timestamp = str(datetime.now())  # return local time
path = "Models/LRModel_" + timestamp
LR_BestModel.save(path)

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
# ### Attribute Information:
#
# Given are the variable name, variable type, the measurement unit and a brief description. The concrete compressive strength is the outcome variable. The order of this listing corresponds to the order of numerals along the rows of the database.
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

# In[1]:


# First let's create our PySpark instance
# import findspark
# findspark.init()

import pyspark  # only run after findspark.init()
from pyspark.sql import SparkSession

# May take awhile locally
spark = SparkSession.builder.appName("Regression_Project").getOrCreate()

cores = spark._jsc.sc().getExecutorMemoryStatus().keySet().size()
print("You are working with", cores, "core(s)")
spark


# **Let's start by reading in our datasets**

# In[2]:


path = "Datasets/"
df = spark.read.csv(path + "Concrete_Data.csv", inferSchema=True, header=True)


# **View data**

# In[3]:


df.limit(6).toPandas()


# **And of course the schema :)**

# In[4]:


df.printSchema()


# **Doulbe check to see if there are any missing values**
#
# Let's go ahead and drop any missing values for the sake of simplicity for this lecture as we have already covered the alternatives in subsequent lectures.

# In[5]:


# drop missing data
drop = df.na.drop()
print("before dropping missings:", df.count())
print("after dropping missings", drop.count())


# All good!

# ## 1. Which features are the strongest predictors of cement strength?
#
# Build your own ML model to figure this one out! This would be good information to give to our client so the sales reps can focus their efforts on certian ingredients to provide recommendations on. For example, if our clients had a customer that was struggling with their cement breaking, we could trouble shoot with them by starting with the factors that we know are important.
#
# So in order to do this, we first need to format our data and create a model!

# ## Format Data
#
# Remember that MLlib requires all input columns of your dataframe to be vectorized. Good thing we created an awesome function to do that in the lectures, so we can simply copy and pasted that here!

# In[6]:


def MLRegressDFPrep(
    df, input_columns, dependent_var, treat_outliers=True, treat_neg_values=True
):

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
        global scalerModel
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


# And apply it

# In[7]:


from pyspark.ml.feature import MinMaxScaler, StringIndexer, VectorAssembler
from pyspark.sql.functions import *
from pyspark.sql.types import *

input_columns = df.columns[:-1]  # all except the last one
dependent_var = df.columns[-1]  # The last column

final_data = MLRegressDFPrep(df, input_columns, dependent_var)
final_data.show(5)


# ## Check for Multicollinearity
#
# Let's make sure we don't have any multicollinearity before we go any further. Remeber the following guidelines for pearson's:
#
# - .00-.19 (very weak)
# - .20-.39 (weak)
# - .40-.59 (moderate)
# - .60-.79 (strong)
# - .80-1.0 (very strong)

# In[8]:


from pyspark.ml.stat import Correlation

pearsonCorr = Correlation.corr(final_data, "features", "pearson").collect()[0][0]
array = pearsonCorr.toArray()


# In[9]:


for item in array:
    print(item[7])


# Looks like the third and fourth features are strongly correlated, but the rest are okay. We may want to consider removing one of the variables in each correlation pair if we decide to use a logistic regression model.

# ## Split dataframe into training and evaluation (test)
#
# I'm going with 70/30 but you can use your own mix if you want.

# In[10]:


train, test = final_data.randomSplit([0.7, 0.3])


# ## Train and test our package of algorithms to see which one works best!
#
# Next, we need to create our model. Let's use our handy dandy function that iterativley runs through all Regression algorithms. I just copy and pasted this from the previous lecture! How easy is that?!

# In[11]:


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
            + "Linear Regression Model Summary without cross validation:"
            + "\033[0m"
        )
        print(" ")
        print("Coefficients: %s" % str(fitModel.coefficients))
        print("Intercept: %s" % str(fitModel.intercept))
        print("")

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

        # Get Model Summary Statistics
        ModelSummary = fitModel.bestModel.summary
        print(
            "Coefficient Standard Errors: "
            + str(ModelSummary.coefficientStandardErrors)
        )
        print(" ")
        print("P Values: " + str(ModelSummary.pValues))  # Last element is the intercept
        print(" ")

        global LR_Pvalues
        LR_Pvalues = ModelSummary.pValues

        # save model
        global LR_BestModel
        LR_BestModel = fitModel.bestModel

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
        print(BestModel.featureImportances)

        # Create Global Variables for feature importances and models
        if Mtype in ("DecisionTreeRegressor"):
            global DT_featureimportances
            DT_featureimportances = BestModel.featureImportances.toArray()
            global DT_BestModel
            DT_BestModel = fitModel.bestModel
        if Mtype in ("GBTRegressor"):
            global GBT_featureimportances
            GBT_featureimportances = BestModel.featureImportances.toArray()
            global GBT_BestModel
            GBT_BestModel = fitModel.bestModel
        if Mtype in ("RandomForestRegressor"):
            global RF_featureimportances
            RF_featureimportances = BestModel.featureImportances.toArray()
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
        result = result.withColumn("Result", result.Result.substr(0, 5))
        return result


# **And now run it!**

# In[12]:


from pyspark.ml.evaluation import *
# Run!
from pyspark.ml.regression import *
# from pyspark.sql import functions as F
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

regressors = [
    LinearRegression(),
    RandomForestRegressor(),
    GBTRegressor(),
    DecisionTreeRegressor(),
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


# **Now for the results!**
#
# Now it's time to query the feature importance lists/arrays that were created above! We can use this information to fine tune our model if we want.

# In[13]:


n = 4

print("Random Forest best features: ", RF_featureimportances.argsort()[-n:][::-1])
print("GBT best features: ", GBT_featureimportances.argsort()[-n:][::-1])
print("Decision Tree best features: ", DT_featureimportances.argsort()[-n:][::-1])
print("Linear Regression best features: ", LR_Pvalues)


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
# The correct answer according to my model is 79.99. Let's see what you get!

# In[22]:


# Manually input our values from above.
values = [(540, 0.0, 0.0, 162, 2.5, 1040, 676, 28)]
# Fetch the column names
column_names = df.columns
column_names = column_names[0:8]
# Map values to column names (always better to soft code :) )
# test = spark.createDataFrame(values,["cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age"])
test = spark.createDataFrame(values, column_names)

# remember that we treated age for right skewness
# so we need to convert the raw value to the transformed value
test = test.withColumn("age", log("age") + 1)

# Transform for a vector
features_list = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]
assembler = VectorAssembler(inputCols=features_list, outputCol="features")
test = assembler.transform(test).select("features")

# rescale each feature to range [min, max].
scaled_test = scalerModel.transform(test)
final_test = scaled_test.select("scaledFeatures")
final_test = final_test.withColumnRenamed("scaledFeatures", "features")

predictions = LR_BestModel.transform(final_test)
predictions.show()


# ## 3. Interact with a user!
#
# Now see if you can ask users to input their own value for Age (keeping all other values the same from the question above) and return a predicted value for the cement stength.
#
# We did not cover this is in the lecture so you'll have to put your thinking cap on. Accepting user input in PySpark works just like it does in traditional Python.
# <br>
#
# val = input("Enter your value: ")

# In[15]:


age_val = input("How old is your cement? ")
values = [(540, 0.0, 0.0, 162, 2.5, 1040, 676, age_val)]
test = spark.createDataFrame(
    values,
    [
        "cement",
        "slag",
        "flyash",
        "water",
        "superplasticizer",
        "coarseaggregate",
        "fineaggregate",
        "age",
    ],
)

# remember that we treated age for right skewness
# so we need to convert the raw value to the transformed value
test = test.withColumn("age", log("age") + 1)

# Transform for a vector
features_list = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]
assembler = VectorAssembler(inputCols=features_list, outputCol="features")
test = assembler.transform(test).select("features")

# rescale each feature to range [min, max].
scaled_test = scalerModel.transform(test)
final_test = scaled_test.select("scaledFeatures")
final_test = final_test.withColumnRenamed("scaledFeatures", "features")

predictions = LR_BestModel.transform(final_test)
response = predictions.select(["prediction"]).collect()
response = response[0][0]
print("Your predicted cement stregth is: ", response)


# ## 4. Make recommendations of optimal values for cement ingredients (our features)
#
# See if you can find the optimal amount of cement to recommend holding the rest of the values from the previous question constant, assuming that the higher the cement strength value the better.

# In[16]:


# First find out the min and max values for cement so we know what grid space to search
df.select("cement", "csMPa").summary("min", "max").show()


# In[17]:


values = [(540, 0.0, 0.0, 162, 2.5, 1040, 676, 28)]
columns = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]
test = spark.createDataFrame(values, columns)

for value in range(50, 700, 10):
    newRow = spark.createDataFrame(
        [(value, 0.0, 0.0, 162, 2.5, 1040, 676, 28)], columns
    )
    test = test.union(newRow)

# remember that we treated age for right skewness
# so we need to convert the raw value to the transformed value
test = test.withColumn("age", log("age") + 1)

# Transform to a vector
features_list = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]
assembler = VectorAssembler(inputCols=features_list, outputCol="features")
# test = assembler.transform(test).select('features')
test = assembler.transform(test)

# rescale each feature to range [min, max].
scaled_test = scalerModel.transform(test)
final_test = scaled_test.withColumnRenamed("features", "oldfeatures")
final_test = final_test.withColumnRenamed("scaledFeatures", "features")

predictions = LR_BestModel.transform(final_test)
predictions.select(["cement", "prediction"]).orderBy(
    predictions["prediction"].desc()
).show(1)

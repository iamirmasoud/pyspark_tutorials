# Machine Learning for Big Data using PySpark with real-world projects

### About this Repo
This repository provides a set of self-study tutorials on Machine Learning for big data using [Apache Spark](https://spark.apache.org/) (PySpark) from basics (Dataframes and SQL) to advanced (Machine Learning Library (MLlib)) topics with practical real-world projects and datasets.

### Preparing the environment
**Note**: I have tested the codes on __Linux__. It can surely be run on Windows and Mac with some little changes.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/pyspark_tutorials.git
cd pyspark_tutorials
```

2. Create (and activate) a new environment, named `spark_env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n spark_env python=3.7.10
	source activate spark_env
	```
	
	At this point your command line should look something like: `(spark_env) <User>:pyspark_tutorials <user>$`. The `(spark_env)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+ and PySpark. You can install  dependencies using:
```
pip install -r requirements.txt
```

4. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd pyspark_tutorials
```

5. Open the directory of notebooks, using the below command. You'll see all files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

6. Once you open any of the project notebooks, make sure you are in the correct `spark_env` environment by clicking `Kernel > Change Kernel > spark_env`.

### Repo Structure:
```
├── 1_Python vs PySpark
│   └── 1_Python vs PySpark
│       ├── Datasets
│       ├── Python vs PySpark [PySpark].ipynb
│       ├── Python vs PySpark [PySpark].py
│       ├── Python vs PySpark [Python].ipynb
│       └── Python vs PySpark [Python].py
├── 2_IO_Filter_SQL
│   ├── 1_Read_Write_and_Validate_Data
│   │   ├── Datasets
│   │   ├── parquet
│   │   ├── partitioned_parquet
│   │   ├── partition_parquet
│   │   ├── part_parquet
│   │   ├── Read_Write_and_Validate_Data_HW.ipynb
│   │   ├── Read_Write_and_Validate_Data_HW.py
│   │   ├── Read_Write_and_Validate_Data_HW_Solutions.ipynb
│   │   ├── Read_Write_and_Validate_Data_HW_Solutions.py
│   │   ├── Read_Write_and_Validate_Data.ipynb
│   │   ├── Read_Write_and_Validate_Data.py
│   │   └── write_test2.csv
│   ├── 2_Search_and_Filter_DataFrames_in_PySpark
│   │   ├── Datasets
│   │   ├── Search and Filter DataFrames in PySpark-HW.ipynb
│   │   ├── Search and Filter DataFrames in PySpark-HW.py
│   │   ├── Search and Filter DataFrames in PySpark-HW-Solutions.ipynb
│   │   ├── Search and Filter DataFrames in PySpark-HW-Solutions.py
│   │   ├── Search and Filter DataFrames in PySpark.ipynb
│   │   └── Search and Filter DataFrames in PySpark.py
│   └── 3_SQL_Options_in_Spark
│       ├── Datasets
│       ├── SQL_Options_in_Spark_HW.ipynb
│       ├── SQL_Options_in_Spark_HW.py
│       ├── SQL_Options_in_Spark_HW_Solutions.ipynb
│       ├── SQL_Options_in_Spark_HW_Solutions.py
│       ├── SQL_Options_in_Spark.ipynb
│       └── SQL_Options_in_Spark.py
├── 3_Manipulation_Aggregation
│   ├── 1_Manipulating_Data_in_DataFrames
│   │   ├── Datasets
│   │   ├── Manipulating_Data_in_DataFrames_HW.ipynb
│   │   ├── Manipulating_Data_in_DataFrames_HW.py
│   │   ├── Manipulating_Data_in_DataFrames_HW_Solutions.ipynb
│   │   ├── Manipulating_Data_in_DataFrames_HW_Solutions.py
│   │   ├── Manipulating_Data_in_DataFrames.ipynb
│   │   └── Manipulating_Data_in_DataFrames.py
│   ├── 2_Aggregating_DataFrames
│   │   ├── Aggregating_DataFrames_in_PySpark_HW.ipynb
│   │   ├── Aggregating_DataFrames_in_PySpark_HW.py
│   │   ├── Aggregating_DataFrames_in_PySpark_HW_Solutions.ipynb
│   │   ├── Aggregating_DataFrames_in_PySpark_HW_Solutions.py
│   │   ├── Aggregating_DataFrames_in_PySpark.ipynb
│   │   ├── Aggregating_DataFrames_in_PySpark.py
│   │   └── Datasets
│   ├── 3_Joining_and_Appending_DataFrames
│   │   ├── Datasets
│   │   ├── Joining_and_Appending_DataFrames_in_PySpark_HW.ipynb
│   │   ├── Joining_and_Appending_DataFrames_in_PySpark_HW.py
│   │   ├── Joining_and_Appending_DataFrames_in_PySpark_HW_Solutions.ipynb
│   │   ├── Joining_and_Appending_DataFrames_in_PySpark_HW_Solutions.py
│   │   ├── Joining_and_Appending_DataFrames_in_PySpark.ipynb
│   │   └── Joining_and_Appending_DataFrames_in_PySpark.py
│   ├── 4_Handling_Missing_Data
│   │   ├── Datasets
│   │   ├── Handling_Missing_Data_in_PySpark_HW.ipynb
│   │   ├── Handling_Missing_Data_in_PySpark_HW.py
│   │   ├── Handling_Missing_Data_in_PySpark_HW_Solutions.ipynb
│   │   ├── Handling_Missing_Data_in_PySpark_HW_Solutions.py
│   │   ├── Handling_Missing_Data_in_PySpark.ipynb
│   │   └── Handling_Missing_Data_in_PySpark.py
│   └── 5_PySpark_Dataframe_Basics
│       ├── Datasets
│       ├── PySpark_Dataframe_Basics_MASTER.ipynb
│       └── PySpark_Dataframe_Basics_MASTER.py
├── 4_Classification_in_PySparks_MLlib
│   ├── 1_Classification_in_PySparks_MLlib
│   │   ├── Classification_in_PySparks_MLlib_with_functions.ipynb
│   │   ├── Classification_in_PySparks_MLlib_with_functions.py
│   │   ├── Classification_in_PySparks_MLlib_without_functions.ipynb
│   │   ├── Classification_in_PySparks_MLlib_without_functions.py
│   │   └── Datasets
│   ├── 2_Classification_in_PySparks_MLlib_with_MLflow
│   │   ├── Classification_in_PySparks_MLlib_with_MLflow.ipynb
│   │   ├── Classification_in_PySparks_MLlib_with_MLflow.py
│   │   └── Datasets
│   └── 3_Classification_in_PySparks_MLlib_Project
│       ├── Classification_in_PySparks_MLlib_Project.ipynb
│       ├── Classification_in_PySparks_MLlib_Project.py
│       ├── Classification_in_PySparks_MLlib_Project_Solution.ipynb
│       ├── Classification_in_PySparks_MLlib_Project_Solution.py
│       └── Datasets
├── 5_NLP_in_Pysparks_MLlib
│   ├── 1_NLP_in_Pysparks_MLlib
│   │   ├── Datasets
│   │   ├── NLP_in_Pysparks_MLlib.ipynb
│   │   └── NLP_in_Pysparks_MLlib.py
│   └── 2_NLP_in_Pysparks_MLlib_Project
│       ├── Datasets
│       ├── NLP_in_Pysparks_MLlib_Project.ipynb
│       ├── NLP_in_Pysparks_MLlib_Project.py
│       ├── NLP_in_Pysparks_MLlib_Project_Solution.ipynb
│       └── NLP_in_Pysparks_MLlib_Project_Solution.py
├── 6_Regression_in_Pysparks_MLlib
│   ├── 1_Regression_in_Pysparks_MLlib
│   │   ├── Datasets
│   │   ├── Regression_in_Pysparks_MLlib_with_functions.ipynb
│   │   ├── Regression_in_Pysparks_MLlib_with_functions.py
│   │   ├── Regression_in_Pysparks_MLlib_without_functions.ipynb
│   │   └── Regression_in_Pysparks_MLlib_without_functions.py
│   └── 2_Regression_in_Pysparks_MLlib_Project
│       ├── Datasets
│       ├── Regression_in_Pysparks_MLlib_Project.ipynb
│       ├── Regression_in_Pysparks_MLlib_Project.py
│       ├── Regression_in_Pysparks_MLlib_Project_Solution.ipynb
│       └── Regression_in_Pysparks_MLlib_Project_Solution.py
├── 7_Unsupervised_Learning_in_Pyspark_MLlib
│   ├── 1_Kmeans_and_Bisecting_Kmeans_in_Pysparks_MLlib
│   │   ├── Datasets
│   │   ├── Kmeans_and_Bisecting_Kmeans_in_Pysparks_MLlib.ipynb
│   │   └── Kmeans_and_Bisecting_Kmeans_in_Pysparks_MLlib.py
│   ├── 2_LDA_in_PySpark_MLlib
│   │   ├── Datasets
│   │   ├── LDA_in_PySpark_MLlib.ipynb
│   │   └── LDA_in_PySpark_MLlib.py
│   ├── 3_GaussuanMixture_in_Pysparks_MLlib
│   │   ├── Datasets
│   │   ├── GaussuanMixture_in_Pysparks_MLlib.ipynb
│   │   └── GaussuanMixture_in_Pysparks_MLlib.py
│   └── 4_Clustering_in_Pysparks_MLlib_Project
│       ├── Clustering_in_Pysparks_MLlib_Project.ipynb
│       ├── Clustering_in_Pysparks_MLlib_Project.py
│       ├── Clustering_in_Pysparks_MLlib_Project_Solution.ipynb
│       ├── Clustering_in_Pysparks_MLlib_Project_Solution.py
│       └── Datasets
├── 8_Frequent_Pattern_Mining_in_PySparks_MLlib
│   ├── 1_Frequent_Pattern_Mining_in_PySparks_MLlib
│   │   ├── Datasets
│   │   ├── Frequent_Pattern_Mining_in_PySparks_MLlib.ipynb
│   │   └── Frequent_Pattern_Mining_in_PySparks_MLlib.py
│   └── 2_Frequent_Pattern_Mining_in_PySparks_MLlib_Project
│       ├── Datasets
│       ├── Frequent_Pattern_Mining_in_PySparks_MLlib_Project.ipynb
│       ├── Frequent_Pattern_Mining_in_PySparks_MLlib_Project.py
│       ├── Frequent_Pattern_Mining_in_PySparks_MLlib_Project_Solution.ipynb
│       └── Frequent_Pattern_Mining_in_PySparks_MLlib_Project_Solution.py
└──────────
``` 
    
## List of real-world projects and datasets
Before run the python scripts and jupyter notebooks of each section, please download necessary datasets for each section from the list below and put them in a directory called `Datasets` next to the scripts. You can find more details about each dataset in the jupyter notebook files.

### 2. IO_Filter_SQL

#### 1_Read_Write_and_Validate_Data
**Datasets:** 
1. [Students Performance in Exams Dataset](https://www.kaggle.com/spscientist/students-performance-in-exams)
2. [PGA Tour 2010-2018 Dataset](https://www.kaggle.com/bradklassen/pga-tour-20102018-data)

#### 2_Search_and_Filter_DataFrames_in_PySpark
**Datasets:** 
1. [FIFA 2019 players and their attributes dataset](https://www.kaggle.com/karangadiya/fifa19) 

#### 3_SQL_Options_in_Spark
**Datasets:** 
1. [Crime dataset for the Police Force Areas of England and Wales](https://www.kaggle.com/r3w0p4/recorded-crime-data-at-police-force-area-level)
2. [Google Play Store Apps Dataset](https://www.kaggle.com/lava18/google-play-store-apps)

### 3. Manipulation_Aggregation

#### 1_Manipulating_Data_in_DataFrames
**Datasets:** 
1. [Daily Trending YouTube Videos Dataset](https://www.kaggle.com/datasnaek/youtube-new#USvideos.csv)
2. [Extracted tweets from all the representatives (latest 200 as of May 17th 2018)](https://www.kaggle.com/kapastor/democratvsrepublicantweets#ExtractedTweets.csv)

#### 2_Aggregating_DataFrames
**Datasets:** 
1. [Activity and Metrics for AirBNB bookers in NYC, NY for 2019 Dataset](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data/data)

#### 3_Joining_and_Appending_DataFrames
**Datasets:** 
1. [UW Madison Courses and Grades 2006-2017 Dataset](https://www.kaggle.com/Madgrades/uw-madison-courses)

#### 4_Handling_Missing_Data
**Datasets:** 
1. [Dataset for aggregate rating of restaurants in Bengaluru India from Zomato](https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants)
2. [New York City Taxi Trip - Hourly Weather Data](https://www.kaggle.com/meinertsen/new-york-city-taxi-trip-hourly-weather-data)

### 4. Classification_in_PySparks_MLlib
#### 1_Classification_in_PySparks_MLlib
**Datasets:** 
1. [Autistic Spectrum Disorder Screening Data for Adult](https://www.kaggle.com/faizunnabi/autism-screening)

#### 2_Classification_in_PySparks_MLlib_with_MLflow
**Datasets:** 
1. [Autistic Spectrum Disorder Screening Data for Adult](https://www.kaggle.com/faizunnabi/autism-screening)

#### 3_Classification_in_PySparks_MLlib_Project
**Project - Genre classification:** 

Have you ever wondered what makes us, humans, able to tell apart two songs of different genres? How we do we inherently know the difference between a pop song and heavy metal? This type of classification may seem easy for us, but it's a very difficult challenge for a computer to do. So the question is, could an automatic genre classification model be possible?
For this project we will be classifying songs based on a number of characteristics into a set of 23 electronic genres. This technology could be used by an application like Pandora to recommend songs to users or just create meaningful channels. Super fun!

**Datasets:**
1. [Electronic Music Features Dataset](https://www.kaggle.com/datasets/caparrini/beatsdataset)




### 5. NLP_in_Pysparks_MLlib
#### 1_NLP_in_Pysparks_MLlib

**Project - Kickstarter Project Success Prediction:** 

Kickstarter is an American public-benefit corporation based in Brooklyn, New York, that maintains a global crowdfunding platform, focused on creativity and merchandising. The company's stated mission is to "help bring creative projects to life". Kickstarter, has reportedly received more than $1.9 billion in pledges from 9.4 million backers to fund 257,000 creative projects, such as films, music, stage shows, comics, journalism, video games, technology and food-related projects.

People who back Kickstarter projects are offered tangible rewards or experiences in exchange for their pledges. This model traces its roots to subscription model of arts patronage, where artists would go directly to their audiences to fund their work.

The goal is to predict if a project will be or not to be able to get the money from their backers.

**Datasets:**
1. [Kickstarter Dataset](https://www.kaggle.com/oscarvilla/kickstarter-nlp)

#### 2_NLP_in_Pysparks_MLlib_Project
**Project - Indeed Real/Fake Job Posting Prediction:** 

Indeed.com has just hired you to create a system that automatically flags suspicious job postings on its website. It has recently seen an influx of fake job postings that is negatively impacting its customer experience. Because of the high volume of job postings it receives every day, their employees don't have the capacity to check every posting, so they would like an automated system that prioritizes which postings to review before deleting it. The final task is to use the attached dataset to create an NLP algorithm which automatically flags suspicious posts for review.

**Datasets:**
1. [Real/Fake Job Posting Prediction Dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)

### 6. Regression_in_Pysparks_MLlib
#### 1_Regression_in_Pysparks_MLlib

**Project - House Price Prediction in California**

**Datasets:**
1. [California Housing Prices Dataset](https://www.kaggle.com/camnugent/california-housing-prices)

#### 2_Regression_in_Pysparks_MLlib_Project
**Project - Cement Strength Prediction based on Ingredients:** 

You have been hired as a consultant to a cement production company who wants to be able to improve their customer experience around a number of areas like being able to provide recommendations to customers on optimal amounts of certain ingredients in the cement making process and perhaps even create an application where users can input their own values and received a predicted cement strength!

**Datasets:**
1. [Yeh Concret Dataset](https://www.kaggle.com/maajdl/yeh-concret-data)

### 7. Unsupervised_Learning_in_Pyspark_MLlib
#### 1_Kmeans_and_Bisecting_Kmeans_in_Pysparks_MLlib
**Project - Customer Segmentation:** 

Use customers data to target marketing efforts! We could use clustering to target similar customer segments. For example, if we do some research about the groups and discover that one is mostly a certain social economic status and purchasing frequency, and offer them a cost savings package that could be beneficial to them. How cool would that be?!

We could also learn a bit more about our clustering by calling on various aggregate statistics for each one of the clusters across each of the variables in our dataframe like this.

**Datasets:**
1. [Customers Credit Card Dataset for Clustering](https://www.kaggle.com/arjunbhasin2013/ccdata)

#### 2_LDA_in_PySpark_MLlib
**Project - Topic Modeling for Cooking Recipes from BBC Good Food:** 

We will be analyzing a collection of Christmas cooking recipes scraped from BBC Good Food. We want to try to discover some additional themes amongst these recipes imagining that we want to create our own website that provides a more intelligent tagging system to recipes that are pulled from multiple data sources.

**Datasets:**
1. [Cooking Recipes from BBC Good Food](https://www.yanlong.app/gjbroughton/christmas-recipes)

#### 3_GaussuanMixture_in_Pysparks_MLlib
**Project - Customer Segmentation based on sales:** 

Indeed.com has just hired you to create a system that automatically flags suspicious job postings on its website. It has recently seen an influx of fake job postings that is negatively impacting its customer experience. Because of the high volume of job postings it receives every day, their employees don't have the capacity to check every posting, so they would like an automated system that prioritizes which postings to review before deleting it. The final task is to use the attached dataset to create an NLP algorithm which automatically flags suspicious posts for review.

**Datasets:**
1. [Customer Sales Data](https://www.kaggle.com/kyanyoga/sample-sales-data)

#### 4_Clustering_in_Pysparks_MLlib_Project
**Project -  University Clustering for the Greater Good:** 

You are a data scientist employed by the ABCDE Foundation, a non-profit organization whose mission is to increase college graduation rates for underprivileged populations. Through advocacy and targeted outreach programs, ABCDE strives to identify and alleviate barriers to educational achievement. ABCDE is driven by the belief that with the right supports, an increase in college attendance and completion rates can be achieved, thereby weakening the grip of the cycles of poverty and social immobility affecting many of our communities. ABCDE is committed to developing a more data driven approach to decision-making. As a prelude to future analyses, ABCDE has requested that you analyze the data to identify clusters of similar colleges and universities.

Your task is to use cluster analysis to identify the groups of characteristically similar schools in the dataset.

**Datasets:**
1. [College Score Dataset](https://data.world/exercises/cluster-analysis-exercise-2)

 
# Handling Big Data for Machine Learning using PySpark with real-world projects

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
    
    
    

 
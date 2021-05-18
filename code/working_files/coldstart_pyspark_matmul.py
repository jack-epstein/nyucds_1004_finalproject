#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Attempt to take in U and V' matrices and get a new R' matrix. Ultimately did not succeed
Usage:
    $ spark-submit --driver-memory 8g --executor-memory 8g code/model/coldstart_pyspark_matmul.py hdfs:/user/jte2004/userFactors_r200 hdfs:/user/jte2004/itemFactors_r200_updated
'''

# Import packages
import sys
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.mllib.linalg import distributed
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, OneHotEncoder
#from pyspark.mllib.recommendation import MatrixFactorizationModel
import time

def main(spark, file_path_users, file_path_items):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_users: parquet files with user latent factors hdfs:/user/jte2004/userFactors_r200
    file_path_items: parquet files with item latent factors hdfs:/user/jte2004/itemFactors_r200_updated
    '''        
    # Loads the parquet files
    user_factors = spark.read.parquet(file_path_users)
    item_factors = spark.read.parquet(file_path_items)
    
    print('User Sample')
    user_factors.printSchema()
    user_factors.limit(3).show()
    print('')
    
    print('Item Sample')
    item_factors.printSchema()
    item_factors.limit(3).show()
    print('')
     
    "https://stackoverflow.com/questions/45789489/how-to-split-a-list-to-multiple-columns-in-pyspark"
    
    '''#test2 = item_factors.select(item_factors.id, item_factors.features[0],item_factors.features[1],item_factors.features[2])
    test2 = item_factors.select(item_factors.id, [item_factors.features[i] for i in range(200)])
    print('new test')
    test2.printSchema()
    print('')'''

    testU = user_factors.select(user_factors.id, user_factors.features[0],user_factors.features[1],user_factors.features[2])
    testV = item_factors.select(item_factors.id, item_factors.features[0],item_factors.features[1],item_factors.features[2])
    
    check = distributed.DistributedMatrix(testU)
    #check = testU.multiply(testV.transpose())
    
    print('matmul test')
    print('rows in U',check.numRows())
    print('')

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('matmul').getOrCreate()
    
    # Get file_path for dataset to analyze
    file_path_users = sys.argv[1]
    file_path_items = sys.argv[2]

    main(spark, file_path_users, file_path_items)

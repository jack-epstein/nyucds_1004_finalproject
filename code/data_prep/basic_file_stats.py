#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to get quick overview of file counts
Usage:
    $ spark-submit basic_file_stats.py <file_path_in> 
    $ spark-submit basic_file_stats.py hdfs:/user/bm106/pub/MSD/cf_train.parquet for the main training file
'''


# Import command line arguments and helper functions
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as func

def main(spark, file_path_in):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in: file to analyze, including HDFS location
    '''    
    
    # Loads the original parquet files
    songs_train = spark.read.parquet(file_path_in)
    
    # Pull the list of users and items from the file
    users = songs_train.select(songs_train.user_id).distinct()
    items = songs_train.select(songs_train.track_id).distinct()
    
    # Check outputs
    print("total rows:",songs_train.count())
    print("total users:",users.count())
    print("total items:",items.count())


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('basicCounts').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in = sys.argv[1]

    main(spark, file_path_in)

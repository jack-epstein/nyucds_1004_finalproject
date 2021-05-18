#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Gets csv with track id and play counts and user plays
Usage:
    $ spark-submit get_plot_csv.py <file_path_in> 
    $ spark-submit code/data_prep/get_plot_csv.py hdfs:/user/bm106/pub/MSD/cf_train.parquet for the main training file
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
    
    songs_train.createOrReplaceTempView('songs_train')
    
    query = """SELECT track_id,
                count(user_id) as user_plays,
                sum(count) as total_plays
            FROM songs_train
            GROUP BY track_id
            ORDER BY user_plays DESC"""
    agg = spark.sql(query)
    
    agg.write.csv('hdfs:/user/jte2004/item_usage.csv')

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('basicCounts').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in = sys.argv[1]

    main(spark, file_path_in)

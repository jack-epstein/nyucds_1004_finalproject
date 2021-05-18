#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script get all items and their hash ID
Usage:
    $ spark-submit code/data_prep/item_hash.py hdfs:/user/bm106/pub/MSD/cf_train.parquet for the main training file
'''

# Import command line arguments, helper functions and pyspark.sql to get the spark session
import sys
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
    items = songs_train.select(songs_train.track_id).distinct()
    
    #create hash id for items
    items = items.withColumn("item_hashId", func.hash("track_id"))
    
    #preview
    print('DF preview')
    print("num hashes:",items.select(items.item_hashId).distinct().count())
    print("num items:",items.select(items.track_id).distinct().count())
    items.limit(10).show()
    
    #write out parquet file
    items.write.parquet('hdfs:/user/jte2004/items_hash.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('itemHash').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in = sys.argv[1]

    main(spark, file_path_in)

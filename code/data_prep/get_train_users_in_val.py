#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to downsample based on items that don't receive enough plays 
Usage:
    $ spark-submit get_train_users_in_val.py <file_path_in_train> <file_path_in_val> <file_path_out>
    $ spark-submit get_train_users_in_val.py hdfs:/user/bm106/pub/MSD/cf_train.parquet hdfs:/user/bm106/pub/MSD/cf_validation.parquet hdfs:/user/jte2004/val_users_in_train.parquet
'''

# Import command line arguments and helper functions
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as func

def main(spark, file_path_in_train, file_path_in_val, file_path_out):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in_train: original training data
    file_path_in_val: original val data
    file_path_out
    '''    
    
    # Loads the original parquet files
    songs_train = spark.read.parquet(file_path_in_train)
    songs_val = spark.read.parquet(file_path_in_val)
    
    #limit the training set to only users also in validation
    songs_train.createOrReplaceTempView('songs_train')
    songs_val.createOrReplaceTempView('songs_val')
    limited = spark.sql('''SELECT 
                            t.user_id,
                            t.count,
                            t.track_id
                        FROM songs_train t INNER JOIN songs_val v ON t.user_id = v.user_id''')
    
    #check new size
    print("total rows in file:",limited.count())
    print("users in new file:",limited.select(limited.user_id).distinct())
    
    
    #hash the user and item ids
    limited = limited.withColumn("user_hashId", func.hash("user_id"))
    limited = limited.withColumn("track_hashId", func.hash("track_id"))
    
    #Save the file
    limited.write.parquet(file_path_out)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('intersectingUsers').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in_train = sys.argv[1]
    file_path_in_val = sys.argv[2]
    file_path_out = sys.argv[3]

    main(spark, file_path_in_train, file_path_in_val, file_path_out)

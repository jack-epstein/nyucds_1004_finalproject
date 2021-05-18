#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to get the 500 most popular songs and output this into a parquet file. NOTE: most popular is determined by number of different people who played a song, not total plays 
Usage:
    $ spark-submit get_popular_items.py <file_path_in>
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
    file_path_in: original training data hdfs:/user/bm106/pub/MSD/cf_train.parquet
    '''    
    
    # Loads the original parquet files
    songs_train = spark.read.parquet(file_path_in)
    
    #get total plays for songs
    songs_train.createOrReplaceTempView('songs_train')
    top_songs = spark.sql('''SELECT 
                            track_id,
                            count(*) as diff_plays
                        FROM songs_train 
                        GROUP BY track_id
                        ORDER BY diff_plays DESC
                        LIMIT 500''')
    
    #confirm correct size
    print("Total items in list:",top_songs.count())
    print('')   
    
    #print a few to check
    print('First 15:')
    top_songs.limit(15).show() 
    
    #confirm correct size
    print("Total items in list:",top_songs.count())
    
    #select only the track_ids and save to a new parquet file
    top_track_ids = top_songs.select(top_songs.track_id)
    top_track_ids.write.parquet('hdfs:/user/jte2004/items_popular.parquet')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('popularSongs').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in = sys.argv[1]

    main(spark, file_path_in)

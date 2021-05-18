#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to create downsampled training dataset based on the most widely listened to items. 
Usage:
    $ spark-submit downsample_items.py <file_path_in> <file_path_out> <cutoff_pct>
    $ spark-submit downsample_items.py hdfs:/user/bm106/pub/MSD/cf_train.parquet hdfs:/user/jte2004/train_downsample_items_50.parquet 0.5 for the main training file
'''

# Import command line arguments and helper functions
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as func

def main(spark, file_path_in, file_path_out, cutoff_pct):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in: original training data hdfs:/user/bm106/pub/MSD/cf_train.parquet 
    file_path_out: new file created hdfs:/user/jte2004/train_downsample_items_50.parquet
    cutoff_pct (float): percent of the items to keep in the new set
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
    print('')
    
    #get total plays for songs
    songs_train.createOrReplaceTempView('songs_train')
    subquery = spark.sql('''SELECT 
                            track_id,
                            sum(count) as play_counts,
                            count(*) as row_totals
                        FROM songs_train 
                        GROUP BY track_id
                        ORDER BY row_totals DESC''')
    
    #get the running total of rows for each item as a percent of total, sorted by frequency
        #IS THERE A WAY NOT TO HARDCODE THE NUMBER? 
    subquery.createOrReplaceTempView('subtable')
    query = spark.sql('''SELECT *,
                        (sum(row_totals) OVER (order by row_totals DESC))/49824519 as row_running_pct
                        FROM subtable''')
    
    #print a few to check
    print('first 5')
    query.limit(5).show() 
    print('')    
    
    #filter this table to only include items that are better than the threshold
    short_list = query.filter(query.row_running_pct <= cutoff_pct)
    
    print("new item count:",short_list.select(short_list.track_id).count())
    print('')
    
    #inner join with the training data and this shortened list
    short_list.createOrReplaceTempView('short_list')
    final_join_query = """SELECT
                            st.user_id,
                            st.count,
                            st.track_id
                          FROM
                            songs_train st INNER JOIN short_list sl ON st.track_id = sl.track_id"""
    songs_train_new = spark.sql(final_join_query)
    
    print('sample of new table')
    songs_train_new.limit(5).show()
    
    #Save the predictions
    songs_train_new.write.parquet(file_path_out)


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('itemTotals').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in = sys.argv[1]
    file_path_out = sys.argv[2]
    cutoff_pct = float(sys.argv[3])

    main(spark, file_path_in, file_path_out, cutoff_pct)

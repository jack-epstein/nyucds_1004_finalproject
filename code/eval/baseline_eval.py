#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to evaluate a baseline popularity model
Usage:
    $ spark-submit --driver-memory 8g --executor-memory 8g code/eval/baseline_eval.py hdfs:/user/jte2004/items_popular.parquet hdfs:/user/jte2004/cf_validation_sort.parquet
'''

# Import packages
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import time

def main(spark, file_path_in_base, file_path_in_val):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in_base: top 500 songs hdfs:/user/jte2004/items_popular.parquet
    file_path_in_val: sorted validation data to use hdfs:/user/jte2004/cf_validation_sort.parquet
    '''        
    # Loads the parquet files
    songs_pop = spark.read.parquet(file_path_in_base)
    songs_val = spark.read.parquet(file_path_in_val)
    
    # Create a hash column as the ID column
    songs_pop = songs_pop.withColumn("track_hashId", func.hash("track_id"))
    songs_val = songs_val.withColumn("user_hashId", func.hash("user_id"))
    songs_val = songs_val.withColumn("track_hashId", func.hash("track_id"))
    
    # Pull the list of users from the val file
    users = songs_val.select(songs_val.user_hashId).distinct() 
    
    #get cross product of users and popular songs
    user_pop_songs = users.crossJoin(songs_pop)
    
    #group by and aggregate to get popular songs into a list
    user_pop_agg = user_pop_songs.groupby("user_hashId").agg(func.collect_list("track_hashId"))
    user_pop_agg = user_pop_agg.withColumnRenamed("collect_list(track_hashId)", "pop_songs")

    
    # Collapse validation file in the same manner - ASSUMES WE READ IN THE SORTED VALIDATION FILE
    songs_val_agg = songs_val.groupby("user_hashId").agg(func.collect_list("track_hashId"))
    songs_val_agg = songs_val_agg.withColumnRenamed("collect_list(track_hashId)", "truth")
   
   
    # EVALUATION
    # Join predictions to ground truth and create default metrics
    start = time.time()
    predictionAndLabels = user_pop_agg.join(songs_val_agg, ["user_hashId"])
    predictionAndLabels = predictionAndLabels.select("pop_songs", "truth")
    predictionAndLabels_rdd = predictionAndLabels.rdd
    
    metrics = RankingMetrics(predictionAndLabels_rdd)
    end = time.time()
    print("Total evaluation time in seconds:",end-start)
    print('')
    
    # Call ranking evaluation metrics
    pr_at = 15
    print('Precision at'+str(pr_at)+':')
    print(metrics.precisionAt(pr_at))
    print('MAP:')
    print(metrics.meanAveragePrecision)
    print('NDCG:')
    print(metrics.ndcgAt(10))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('BaselinePopularity').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in_base = sys.argv[1]
    file_path_in_val = sys.argv[2]

    main(spark, file_path_in_base, file_path_in_val)

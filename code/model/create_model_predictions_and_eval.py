#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to run and evaluate an ALS model the train data set.
Usage:
    $ spark-submit --driver-memory 8g --executor-memory 8g code/model/create_model_predictions_and_eval.py <file_path_in> <file_path_in_val> <max_iter> <reg> <rk>
'''

# Import packages
import sys
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
import time

def main(spark, file_path_in_train, file_path_in_val, max_iter, reg, rk):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in_train: training data to use hdfs:/user/jte2004/train_downsample_items_{size}.parquet
    file_path_in_val: validation data to use hdfs:/user/jte2004/cf_validation_sort.parquet
    maxIter (int): number of iterations to fit ALS model
    reg (float): regularization parameter
    rk (int): rank of matrices U and V in model
    '''        
    # Loads the parquet files
    songs_train = spark.read.parquet(file_path_in_train)
    songs_val = spark.read.parquet(file_path_in_val)
    
    # Create a hash column as the ID column
    songs_train = songs_train.withColumn("user_hashId", func.hash("user_id"))
    songs_train = songs_train.withColumn("track_hashId", func.hash("track_id"))
    songs_val = songs_val.withColumn("user_hashId", func.hash("user_id"))
    songs_val = songs_val.withColumn("track_hashId", func.hash("track_id"))
    
    #check to ensure correct file size (can delete after)
    print("total rows:",songs_train.count())
    print("% of total:",songs_train.count()/49824519)
    print('')
    
    #1. MODELING
    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    start = time.time()
    als = ALS(maxIter=max_iter, regParam=reg, rank=rk, userCol="user_hashId", itemCol="track_hashId", ratingCol="count",
              coldStartStrategy="drop", implicitPrefs=True, seed=1234)
    model = als.fit(songs_train)
    end = time.time()
    print("Total model time in seconds:",end-start)
    print('')
    
    # Pull the list of users from the val file
    users = songs_val.select(songs_val.user_hashId).distinct()
    
    
    
    #2. PREDICTIONS
    start = time.time()
    #generate predictions for top 500 songs
    songRecs = model.recommendForUserSubset(users, 500)
    
    # Transform the preds into a ranking
    # By default, the recommendations column contains dataframe objects with track_hashId and score columns
    songRecs = songRecs.withColumn("recommendations_ranked", songRecs.recommendations.track_hashId)
    songRecs = songRecs.select("user_hashId", "recommendations_ranked")   
    end = time.time()
    print("Total prediction time in seconds:",end-start)
    print('')
    
    #show a few samples of the song recs #LIKELY DELETE THIS
    '''songRecs.createOrReplaceTempView('songRecs')
    query = spark.sql('SELECT * FROM songRecs limit 10')
    query.show()'''   
        
        
        
        
    # 3. EVALUATION
    # Collapse validation file
        #Assuming the validation file is aleady sorted
    #songs_val = songs_val.sort(["user_hashId", "count"], ascending=[1, 0])
    songs_val_agg = songs_val.groupby("user_hashId").agg(func.collect_list("track_hashId"))
    songs_val_agg = songs_val_agg.withColumnRenamed("collect_list(track_hashId)", "truth")
    
    # Join predictions to ground truth and create default metrics
    start = time.time()
    predictionAndLabels = songRecs.join(songs_val_agg, ["user_hashId"])
    predictionAndLabels = predictionAndLabels.select("recommendations_ranked", "truth")
    
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
    spark = SparkSession.builder.appName('ALSModel').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in_train = sys.argv[1]
    file_path_in_val = sys.argv[2]
    max_iter = int(sys.argv[3])
    reg = float(sys.argv[4])
    rk = int(sys.argv[5])

    main(spark, file_path_in_train, file_path_in_val, max_iter, reg, rk)

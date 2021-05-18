#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Attempt to complete the entire cold start model in one spark script -- we ultiamtely chose to run parts locally
Usage:
    $ spark-submit --driver-memory 8g --executor-memory 8g code/model/create_model_predictions_and_eval_coldstart_working.py <file_path_in> <file_path_in_val> <file_path_in_meta> <max_iter> <reg> <rk>
'''

# Import packages
import sys
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, OneHotEncoder
#from pyspark.mllib.recommendation import MatrixFactorizationModel
import time

def main(spark, file_path_in_train, file_path_in_val, max_iter, reg, rk):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in_train: training data to use hdfs:/user/jte2004/train_downsample_items_{size}.parquet
    file_path_in_val: validation data to use hdfs:/user/jte2004/cf_validation_sort.parquet
    NEED TO LEAVE IN SPACE TO PULL IN THE LATENT FACTOR MATRIX
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
    
    
    #1. DOWNSAMPLING
    #Randomly downsample the items
    items = songs_train.select(songs_train.track_hashId).distinct()
    items_down = items.sample(fraction = 0.5, seed = 0)
    
    print('before size:',songs_train.count())
    #filter out the training data not included in the random list
    songs_train = songs_train.join(items_down, ['track_hashId'])
    print('after size:',songs_train.count())
    print('')
    
    
    #2. MODELING
    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    start = time.time()
    als = ALS(maxIter=max_iter, regParam=reg, rank=rk, userCol="user_hashId", itemCol="track_hashId", ratingCol="count",
              coldStartStrategy="drop", implicitPrefs=True, seed=0)
    model = als.fit(songs_train)
    end = time.time()
    print("Total model time in seconds:",end-start)
    print('')
    
    # Pull the list of users from the val file
    users = songs_val.select(songs_val.user_hashId).distinct().count()
    
    latent = model.itemFactors
    print('latent factor df')
    latent.limit(10).show()
    
    written = model.write() 
    
    print('')
    print(type(written))
    
    
    #HOW CAN WE SET UP A LINEAR REGRESSION MODEL TO PREDICT EACH OF THE LATENT FACTORS
        #WILL NEED k DIFFERENT REGRESSION MODELS
        #each inputs the same features of metadata and outputs each of the latent features
        
    #FOR ITEMS NOT IN THE DATASET, HOW DO WE THEN INCLUDE THEM IN A PREDICTION?
    
    #3. PREDICTIONS
    '''start = time.time()
    #generate predictions for top 500 songs
    songRecs = model.recommendForUserSubset(users, 500)
    
    # Transform the preds into a ranking
    # By default, the recommendations column contains dataframe objects with track_hashId and score columns
    songRecs = songRecs.withColumn("recommendations_ranked", songRecs.recommendations.track_hashId)
    songRecs = songRecs.select("user_hashId", "recommendations_ranked")   
    end = time.time()
    print("Total prediction time in seconds:",end-start)
    print('')
        
        
    # 4. EVALUATION
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
    print(metrics.ndcgAt(10))'''

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('ALSModel_ColdStart').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in_train = sys.argv[1]
    file_path_in_val = sys.argv[2]
    max_iter = int(sys.argv[3])
    reg = float(sys.argv[4])
    rk = int(sys.argv[5])

    main(spark, file_path_in_train, file_path_in_val, max_iter, reg, rk)

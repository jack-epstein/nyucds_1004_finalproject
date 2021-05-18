#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Script to run and evaluate an ALS model the train data set.
Usage:
    $ spark-submit --driver-memory 8g --executor-memory 8g code/model/create_model_predictions_and_eval_hyperparams.py <file_path_in> <file_path_in_val> <max_iter>
'''

# Import packages
import sys
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS
import time

#DELETE THESE: WHAT ELSE NEEDS TO HAPPEN
    # UPDATE THE STRATEGIC DOWNSAMPLING TO MAKE NEW PARQUET FILES (NO CHANGES NEEDED HERE)
    # CHECK OUT SARAS VALIDATION SORTING TO CHANGE WHICH FILE I CAN READ IN HERE. THEN UPDATE VAL AGG STEPS

def main(spark, file_path_in_train, file_path_in_val, max_iter):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    file_path_in_train: training data to use hdfs:/user/jte2004/train_downsample_items_{size}.parquet
    file_path_in_val: validation data to use hdfs:/user/jte2004/cf_validation_sort.parquet
    maxIter (int): number of iterations to fit ALS model
    '''        
    # Loads the parquet files
    songs_train = spark.read.parquet(file_path_in_train)
    songs_val = spark.read.parquet(file_path_in_val)
    
    # Create a hash column as the ID column
    songs_train = songs_train.withColumn("user_hashId", func.hash("user_id"))
    songs_train = songs_train.withColumn("track_hashId", func.hash("track_id"))
    songs_val = songs_val.withColumn("user_hashId", func.hash("user_id"))
    songs_val = songs_val.withColumn("track_hashId", func.hash("track_id"))
    
    # Pull the list of users from the val file
    users = songs_val.select(songs_val.user_hashId).distinct()
      
    #set up
    regs = [0.001,0.01,0.1,1]
    ranks = [10,20,50,100,200]
    
    #set up dictionary to store MAP
    maps = {}
        
    #add in timers here and comments
    for rk in ranks: 
        
        #start a list for 
        maps[rk] = []
        
        for reg in regs:
            
            start = time.time()
            
            #fit ALS model
            als = ALS(maxIter=max_iter, regParam=reg, rank=rk, userCol="user_hashId", itemCol="track_hashId", ratingCol="count",
              coldStartStrategy="drop", implicitPrefs=True, seed=1234)
            model = als.fit(songs_train)
            
            #get predictions and transform the preds into a ranking
            songRecs = model.recommendForUserSubset(users, 500)
            songRecs = songRecs.withColumn("recommendations_ranked", songRecs.recommendations.track_hashId)
            songRecs = songRecs.select("user_hashId", "recommendations_ranked")   
        
            # Collapse validation file
                #Assuming the validation file is aleady sorted
                #songs_val = songs_val.sort(["user_hashId", "count"], ascending=[1, 0])
            songs_val_agg = songs_val.groupby("user_hashId").agg(func.collect_list("track_hashId"))
            songs_val_agg = songs_val_agg.withColumnRenamed("collect_list(track_hashId)", "truth")
            
            #join predictions and truth into one rdd
            predictionAndLabels = songRecs.join(songs_val_agg, ["user_hashId"])
            predictionAndLabels = predictionAndLabels.select("recommendations_ranked", "truth")
            predictionAndLabels_rdd = predictionAndLabels.rdd
            
            #get metrics
            metrics = RankingMetrics(predictionAndLabels_rdd)
            MAP = metrics.meanAveragePrecision
            print('')
            print('Rank:',rk)
            print('Regularization:',reg)
            print('Mean Average Precision:',MAP)
            
            #save the MAP into the dict
            maps[rk].append(MAP)
            end = time.time()
        
        print('')
        print('Total time for loop of rank:',end-start)
        print('------------------------------------')
        print('')
        
    print(maps)
        

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('ALSModel_hyperparams').getOrCreate()

    # Get file_path for dataset to analyze
    file_path_in_train = sys.argv[1]
    file_path_in_val = sys.argv[2]
    max_iter = int(sys.argv[3])

    main(spark, file_path_in_train, file_path_in_val, max_iter)

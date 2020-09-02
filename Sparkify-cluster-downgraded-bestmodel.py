# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, count, when, col, desc, udf, col, sort_array, asc, \
                                  avg, from_unixtime, split, min, max, round, lit, mean
from pyspark.sql.functions import sum as Fsum
from pyspark.sql.functions import row_number
from pyspark.sql.window import Window

from pyspark.sql.types import IntegerType, TimestampType
import datetime
from pyspark.sql.functions import to_date, year, month, dayofmonth, dayofweek, hour, date_format, substring

from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

import numpy as np

# ---------------
# Functions
# ---------------

# Function that returns all users of a specified churn group
def get_users(churn):
    return data.where(data.churn == 1).select('userId').dropDuplicates()

# Count page logs
def page_count(page):
    return page_data \
        .where(data.page == page) \
        .groupby('userId') \
        .count() \
        .select('userId', col('count').alias(page.replace(' ', '') + 'Count'))

# Average page count per session hour
def page_session_hour(page):
    return page_data \
        .where(data.page == page) \
        .join(session_hours, ['userId', 'sessionId'], 'inner') \
        .groupby( 'userId', 'sessionId', 'sessionHours') \
        .agg((count('userId')/col('sessionHours')).alias('avgPerSession')) \
        .groupby('userId') \
        .agg(avg('avgPerSession').alias('avg')) \
        .select('userId', col('avg').alias(page.replace(' ', '') + 'PerSessionHour'))

# Average page count per hour
def page_hour(page):
    return page_data \
        .where(data.page == page) \
        .join(user_hours, 'userId', 'inner') \
        .groupby('userId', 'hours') \
        .agg((count('userId')/col('hours')).alias('avg')) \
        .select('userId', col('avg').alias(page.replace(' ', '') + 'PerHour'))

# Average page count per day
def page_day(page):
    return page_data \
        .where(data.page == page) \
        .groupby('userId', 'date') \
        .count() \
        .groupby('userId') \
        .agg(avg('count').alias(page.replace(' ', '') + 'PerDay'))

# Return selected features without label  
def get_features(data):
    return data.drop('label').columns

# Split data into training and testing subset
def split_data(data, seed=0):
    train, test = data.randomSplit([0.8, 0.2], seed=seed);
    return train, test

# Create ML pipeline
def create_pipeline(data, classifier, scaler):
    features = get_features(data)
    assembler = VectorAssembler(inputCols=features, outputCol='NumFeatures')
    pipeline = Pipeline(stages=[assembler, scaler, classifier]);
    return pipeline

# Fit and evaluate model
def fit_eval_model(data, classifier, seed=0):
    train, test = split_data(data, seed)
    scaler = StandardScaler(inputCol='NumFeatures', outputCol='features')
    pipeline = create_pipeline(data, classifier, scaler)
    model = pipeline.fit(train)
    calc_metrics(data, model, test)   
    return model

# Calc and store metrics (F1, accuracy, weigted precision, weighted recall, AUC)
# source: https://stackoverflow.com/questions/60772315/how-to-evaluate-a-classifier-with-apache-spark-2-4-5-and-pyspark-python
def calc_metrics(data, model, test):

    # Create both evaluators
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')
    evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction', metricName='areaUnderROC')

    # Make predicitons
    prediction = model.transform(test).select('label', 'prediction')

    # Get metrics
    accuracy = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'accuracy'})
    f1 = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'f1'})
    weightedPrecision = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'weightedPrecision'})
    weightedRecall = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'weightedRecall'})
    auc = evaluator.evaluate(prediction)

    # Write metrics to S3
    columns1 = ['f1', 'accuracy', 'weightedPrecision', 'weightedRecall', 'AUC']
    values1 = [f1, accuracy, weightedPrecision, weightedRecall, auc]
    rdd = spark.sparkContext.parallelize([values1])
    df1 = rdd.toDF(columns1)
    df1.write.mode('overwrite').json('s3n://amosvoron-sparkify/rfc-downgraded-metrics-11h.json')

    # Write feature importances to S3
    values2 = []
    features = list(get_features(data))
    importances = list(model.stages[-1].featureImportances)
    for x in list(zip(features, importances)):
        values2.append([x[0], float(x[1])])
    columns2 = ['feature', 'importance']
    rdd = spark.sparkContext.parallelize(values2)
    df2 = rdd.toDF(columns2) 
    df2.write.mode('overwrite').json('s3n://amosvoron-sparkify/rfc-downgraded-feature-importances-11h.json')


if __name__ == "__main__":

    # ---------------
    # Create session
    # Load data
    # Prepare data
    # ---------------

    spark = SparkSession \
        .builder \
        .appName("Sparkify-Downgraded-22") \
        .getOrCreate()

    # Set time parser policy
    spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

    # Read full sparkify dataset
    event_data = "s3n://udacity-dsnd/sparkify/sparkify_event_data.json"
    data = spark.read.json(event_data)

    # Remove rows with missing users
    data = data.where(~((col('userId').isNull()) | (col('userId') == '')))

    # Exclude non-relevant columns
    data = data.drop('firstName')
    data = data.drop('lastName')

    # Add tsDate and date column
    data = data.withColumn('tsDate', (col('ts') / 1000.0).cast(TimestampType()))
    data = data.withColumn('date', date_format(col('tsDate'), 'yyyy-MM-dd').alias('date').cast('date'))

    # Label churned users using Cancellation Confirmation event
    query_churn_by_cc = data.where(data.page == 'Submit Downgrade')
    canceled = query_churn_by_cc.select('userId').dropDuplicates().select('userId')
    canceled_uids = [row.userId for row in canceled.collect()];
    set_churn = udf(lambda x: 1 if x in canceled_uids else 0, IntegerType())
    data = data.withColumn('churn', set_churn('userId'))

    # Add [userRowId] column that assigns a 1-based index to every user's log ordered by [ts]
    w =  Window.partitionBy(data.userId).orderBy('ts', 'itemInSession')
    data = data.withColumn('userRowId', row_number().over(w))

    # Add [userRowDescId] column that assigns a 1-based index to every user's log ordered by [ts] descending.
    w =  Window.partitionBy(data.userId).orderBy(col('ts').desc(), col('itemInSession').desc())
    data = data.withColumn('userRowDescId', row_number().over(w))

    # Add last level column
    last_levels = dict()
    for row in data.where(data.userRowDescId == 1).select('userId', 'level').collect():
	    last_levels[row.userId] = row.level
    get_level = udf(lambda userId: last_levels[userId])
    data = data.withColumn('lastLevel', get_level('userId'))

    # Prepare labels
    labels = data.select(col('churn').alias('label'), 'userId').dropDuplicates()

    # ---------------
    # Queries
    # ---------------

    # All unique users
    users = data.select('userId').dropDuplicates()

    # Pages without churn definition events
    page_data = data.where(~data.page.isin(['Cancel', 'Cancellation Confirmation', 'Downgrade', 'Submit Downgrade', 'Upgrade', 'Submit Upgrade'])) \
        .select('page', 'userId', 'sessionId', 'ts', 'date')

    # Calc session duration (in hours)
    session_hours = page_data \
        .groupby('userId', 'sessionId') \
        .agg(((max('ts') - min('ts'))/1000/3600).alias('sessionHours'))

    # User interactions duration per user (in hours)
    user_hours = page_data \
        .groupby('userId', 'sessionId') \
        .agg(((max('ts') - min('ts'))/1000/3600).alias('sessionHours')) \
        .groupby('userId') \
        .agg(Fsum('sessionHours').alias('hours'))    

    # ---------------
    # Features
    # ---------------

    f_Gender = data \
        .select('userId', 'gender') \
        .dropDuplicates() \
        .replace(['M', 'F'], ['0', '1'], 'gender') \
        .select('userId', col('gender').cast('int').alias('Gender'))

    f_LastLevel = data \
        .select('userId', 'lastLevel') \
        .dropDuplicates() \
        .replace(['free', 'paid'], ['0', '1'], 'lastLevel') \
        .select('userId', col('lastLevel').cast('int').alias('LastLevel'))

    f_LogCount = data \
        .groupby('userId') \
        .agg(count('userId').alias('LogCount'))

    f_SongCount = data \
        .where(data.page == 'NextSong') \
        .groupby('userId') \
        .agg(count('userId').alias('SongCount'))

    f_NonSongCount = data \
        .where(data.page != 'NextSong') \
        .groupby('userId') \
        .agg(count('userId').alias('NonSongCount'))

    f_AboutCount = page_count('About')

    f_ThumbsUpCount = page_count('Thumbs Up')

    f_ThumbsDownCount = page_count('Thumbs Down')

    f_RollAdvertCount = page_count('Roll Advert')

    f_AddFriendCount = page_count('Add Friend')

    f_AddToPlaylistCount = page_count('Add to Playlist')

    f_LogoutCount = page_count('Logout')    

    f_HomeCount = page_count('Home')

    f_SettingsCount = page_count('Settings')  

    f_SaveSettingsCount = page_count('Save Settings')    

    f_SessionCount = data \
        .select('userId', 'sessionId') \
        .dropDuplicates() \
        .groupby('userId') \
        .agg(count('userId').alias('SessionCount'))

    f_AvgSessionLength = data \
       .groupby('userId', 'sessionId') \
       .agg(((max('ts') - min('ts'))/1000).alias('sessionLength')) \
       .groupby('userId') \
       .agg(avg('sessionLength').alias('AvgSessionLength')) \

    f_AvgSessionGap = data \
        .groupby('userId', 'sessionId') \
        .agg(min('ts').alias('startTime'), max('ts').alias('endTime')) \
        .groupby('userId') \
        .agg(count('userId').alias('sessionCount'), \
            ((max('endTime') - min('startTime'))/1000).alias('observationPeriodTime'), \
            (Fsum(col('endTime') - col('startTime'))/1000).alias('totalSessionTime')) \
        .where(col('sessionCount') > 1) \
        .join(users, 'userId', 'outer') \
        .fillna(0) \
        .select('userId', \
            (col('observationPeriodTime') - col('totalSessionTime')/(col('sessionCount') - 1)).alias('AvgSessionGap'))

    f_AboutPerSessionHour = page_session_hour('About')

    f_ErrorPerSessionHour = page_session_hour('Error')

    f_SettingsPerSessionHour = page_session_hour('Settings')

    f_SaveSettingsPerSessionHour = page_session_hour('Save Settings')

    f_LogoutPerSessionHour = page_session_hour('Logout')

    f_AboutPerHour = page_hour('About')

    f_ErrorPerHour = page_hour('Error')    

    f_RollAdvertPerHour = page_hour('Roll Advert')

    f_ThumbsDownPerHour = page_hour('Thumbs Down')

    f_SubmitUpgradePerHour = page_hour('Submit Upgrade')

    f_SaveSettingsPerHour = page_hour('Save Settings')

    f_HomePerHour = page_hour('Home')

    f_LogoutPerHour = page_hour('Logout')

    f_SettingsPerHour = page_hour('Settings')

    f_SessionsPerDay = data \
        .select('userId', 'date', 'sessionId') \
        .dropDuplicates() \
        .groupby('userId', 'date') \
        .count() \
        .groupby('userId') \
        .agg(avg('count').alias('SessionsPerDay'))

    f_AddFriendPerDay = page_day('Add Friend')

    f_RollAdvertPerDay = page_day('Roll Advert')

    f_ThumbsDownPerDay = page_day('Thumbs Down')

    f_ThumbsUpPerDay = page_day('Thumbs Up')

    f_TotalSongLength = data \
        .where(data.page == 'NextSong') \
        .select('userId', 'length') \
        .groupby('userId') \
        .agg(Fsum('length').alias('TotalSongLength'))

    f_UniqueSongCount = data \
        .where(data.page == 'NextSong') \
        .select('userId', 'song') \
        .dropDuplicates() \
        .groupby('userId') \
        .agg(count('userId').alias('UniqueSongCount'))

    f_UniqueSongShare = data \
        .where(data.page == 'NextSong') \
        .select('userId', 'song') \
        .dropDuplicates() \
        .groupby('userId') \
        .count() \
        .join(f_SongCount, on = ['userId'], how = 'inner') \
        .select('userId', (col('count')/col('SongCount')).alias('UniqueSongShare')) 

    # ---------------
    # Fit model
    # ---------------

    # Best features (11)
    data = labels.join(f_HomePerHour, 'userId', 'outer') \
        .join(f_RollAdvertPerHour, 'userId', 'outer') \
        .join(f_SongCount, 'userId', 'outer') \
        .join(f_NonSongCount, 'userId', 'outer') \
        .join(f_TotalSongLength, 'userId', 'outer') \
        .join(f_UniqueSongCount, 'userId', 'outer') \
        .join(f_LogCount, 'userId', 'outer') \
        .join(f_AddFriendCount, 'userId', 'outer') \
        .join(f_LogoutCount, 'userId', 'outer') \
        .join(f_RollAdvertCount, 'userId', 'outer') \
        .join(f_ThumbsDownCount, 'userId', 'outer') \
        .drop('userId') \
        .fillna(0)

    classifier = RandomForestClassifier(seed=0, maxDepth=6, numTrees=50)

    fit_eval_model(data, classifier, seed=0)

    # ---------------
    # Terminate job
    # ---------------

    spark.stop()

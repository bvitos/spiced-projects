#! python3
# etl.py - ETL Docker script: Reads tweets from mongodb, performs sentiment analysis 
# and stores results (incl. timestamp of analysis) in postgres

import pymongo
from sqlalchemy import create_engine
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import logging
import os

time.sleep(5)  # safety check

pguser = os.environ.get('POSTGRES_USER')               # postgres credentials: using environment vars stored in hidden file
pgpassword = os.environ.get('POSTGRES_PASSWORD')
pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres') # pg connect

client = pymongo.MongoClient('mongodb')      # mongo connect
db = client.test
tweetsdb = db.tweets

oldid = ''
while True:                                      # keeps on checking for arrival of tweets with new ID
    latesttweet = list(tweetsdb.find().limit(1).sort([( '$natural', -1 )] ))
    newid = str(latesttweet[0]['_id'])            # read tweet ID
    if newid != oldid:
        newtweet = latesttweet[0]['tweet']        # read tweet text
        if newtweet[0:3] == 'RT ':                              # remove RT prefix if present
            newtweet = newtweet[3:]
        oldid = newid
        now = datetime.now()
        s = SentimentIntensityAnalyzer()
        sentscore = s.polarity_scores(newtweet)['compound']                # get sentiment score
        logging.critical(f'\n\nFORWARDING TO POSTGRES: {newtweet}\n')
        logging.critical(f'WITH SENTIMENT SCORE: {sentscore}\n\n')
        data = [[newid,newtweet,now.strftime('%Y-%m-%d %H:%M:%S'),sentscore]]   # database row to be stored in postgres
        tweets_new = pd.DataFrame(data, columns=['id','tweet','timestamp','sentimentscore'])
        tweets_new.set_index('id', inplace=True)
        tweets_new.to_sql('tweets', pg, if_exists='append', method='multi', chunksize=1000)  # exporting to postgres
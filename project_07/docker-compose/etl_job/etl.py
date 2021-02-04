#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL Docker script: Reads tweets from MongoDB, performs VADER sentiment analysis 
and stores results (incl. current timestamp) in Postgres
"""

import pymongo
from sqlalchemy import create_engine
import time
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import logging
import os

if __name__ == '__main__':
    pguser = os.environ.get('POSTGRES_USER')               # postgres credentials imported from .env file
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres') # pg connect
    client = pymongo.MongoClient('mongodb')      # mongo connect
    db = client.tweets
    tweetsdb = db.tmp
    tweetsdb.drop()
    tweetsdb = db.tmp                           # create empty "tmp" collection to be populated by get_tweets.py
                                                # note: tweets are stored in the "backup" mongo collections

    while True:                                      # keep on checking for arrival of new tweets in "tmp"
        latesttweet = list(tweetsdb.find().limit(1).sort([( '$natural', -1 )] ))
        if len(latesttweet) > 0:                    # new tweet found:
            tweetsdb.remove({"_id":latesttweet[0]['_id']})          # remove already processed tweet from db
            newid = str(latesttweet[0]['_id'])                      # read tweet ID
            newtweet = latesttweet[0]['tweet']                      # read tweet text
            if newtweet[0:3] == 'RT ':                              # remove RT prefix if present
                newtweet = newtweet[3:]
            now = datetime.now().replace(microsecond=0)
            s = SentimentIntensityAnalyzer()
            sentscore = s.polarity_scores(newtweet)['compound']                # get sentiment score
            logging.critical(f'\n\nFORWARDING TO POSTGRES: {newtweet}\n')
            logging.critical(f'WITH SENTIMENT SCORE: {sentscore}\n\n')
            data = [[newid,newtweet,now,sentscore]]   # create database row to be stored in postgres
            tweets_new = pd.DataFrame(data, columns=['id','tweet','timestamp','sentimentscore'])
            tweets_new.set_index('id', inplace=True)
            tweets_new.to_sql('tweets', pg, if_exists='append', method='multi', chunksize=1000)  # export to postgres
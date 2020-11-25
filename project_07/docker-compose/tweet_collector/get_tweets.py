#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intercepts tweets realtime and stores them in Mongo database
Imports tweet keywords from config file andd applies them realtime
"""

from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
import json
import logging
import time
import pymongo
import os
import pandas as pd

def authenticate():
    """ Twitter Authentication. Credentials imported from .env file
    """
    auth = OAuthHandler(os.environ.get('API_KEY'), os.environ.get('API_SECRET'))
    auth.set_access_token(os.environ.get('ACCESS_TOKEN'), os.environ.get('ACCESS_TOKEN_SECRET'))
    return auth

class TwitterListener(StreamListener):

    def on_data(self, data):

        """ Intercepting tweets real-time and adding them 
        to Mongo database"""

        t = json.loads(data) # python dictionary

        tweet = {
        'text': t['text'],
        'username': t['user']['screen_name'],
        'followers_count': t['user']['followers_count']
        }

        tweetimport = tweet["text"] + " ... POSTED BY: " + tweet["username"]    # tweet details to be stored

        logging.critical(f"\n\n\Imported tweet: {tweetimport}\n\n")             # log it

        global collectionbak
        collectionbak.insert_one({"tweet": tweetimport})                           # Mongo importing
        global collection
        collection.insert_one({"tweet": tweetimport})                           # Mongo importing
        

    def on_error(self, status):

        if status == 420:
            logging.critical(status)                                            # log error
            return False

if __name__ == '__main__':

    client = pymongo.MongoClient(host='mongodb', port=27017)                    # set up Mongo
    db = client.tweets
    collection = db.tmp                 # this is used for ETL (to be emptied upon use)
    collectionbak = db.backup           # this is used for backing up the tweets

    auth = authenticate()                                                       # set up Twitter listener
    listener = TwitterListener()
    stream = []
    stream.append(Stream(auth, listener))
    configdata=pd.read_csv('../config/config.csv')
    keywordslist = configdata.iloc[0,1]
    keywords = keywordslist.split(',')
    stream[len(stream)-1].filter(track= keywords, languages=['en'], is_async=True) # initiate Twitter stream

    while True:
        time.sleep(5)                       # Check for new Twitter keyword settings every 5 secs
        configdata=pd.read_csv('../config/config.csv')
        keywordslistnew = configdata.iloc[0,1]
        if keywordslistnew != keywordslist:
            keywordsnew = keywordslistnew.split(',')
            keywordslist = keywordslistnew
            logging.critical(f"\n\nSETTINGS CHANGE .... Listening to new tweet keywords: {keywordslistnew}. \n\n")             # log it
            logging.critical(keywordslistnew)
            stream[len(stream)-1].disconnect()                 # stop previous stream
            stream.append(Stream(auth, listener))
            stream[len(stream)-1].filter(track = keywordsnew, languages=['en'], is_async=True) # initiate new one

#! python3
# slackbot.py - Slackbot script: Reads tweets from postgres, post them on Slack based on their sentiment score

from sqlalchemy import create_engine
import time
from datetime import datetime
import pandas as pd
import logging
import os
import requests

time.sleep(5)  # safety check

pguser = os.environ.get('POSTGRES_USER')               # Import Postgres credentials form env file
pgpassword = os.environ.get('POSTGRES_PASSWORD')
pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres') # pg connect

webhook_url = os.environ.get('SLACK_WEBHOOK')           # Import Slack webhook from env file

now = datetime.now()
olddatetime = now.strftime('%Y-%m-%d %H:%M')
oldminute = 60
configdata=pd.read_csv('../config/config.csv')
botswitch = configdata.iloc[0,0]
while True:
    now = datetime.now()
    minutenow = now.minute
    configdata=pd.read_csv('../config/config.csv')      # on-the-fly Slackbot switch using config file
    botrunning = configdata.iloc[0,0]
    if (botswitch != botrunning):
        botswitch = botrunning
        logging.critical(f'\n\nSLACKBOT SWITCHED TO STATE: {botrunning}')
    if (minutenow != oldminute) and (botrunning == 0):
        oldminute = minutenow
        olddatetime = now.strftime('%Y-%m-%d %H:%M')            	
    if (minutenow != oldminute) and (botrunning == 1):                       # check tweets in postgres in one-minute intervals
#        out = pd.read_sql_query("SELECT * FROM tweets WHERE timestamp LIKE (?);", pg, params = [olddatetime])   # first, dump the last 100 tweets into a dataframe 
                                                                                             # so we don't have to search through the whole db all the time)                

        result = pd.read_sql_query("SELECT * FROM tweets ORDER BY id DESC LIMIT 100;", pg)   # first, dump the last 100 tweets into a dataframe 
                                                                                             # so we don't have to search through the whole db all the time)                
        out = result[result["timestamp"].str.contains(olddatetime)]   # filter last minute's tweets
        happytweet = out[out["sentimentscore"] == out["sentimentscore"].max()]
        bestnews = happytweet["tweet"].values[0]
        bestscore = happytweet["sentimentscore"].values[0]
        sadtweet = out[out["sentimentscore"] == out["sentimentscore"].min()]
        worstnews = sadtweet["tweet"].values[0]
        worstscore = sadtweet["sentimentscore"].values[0]        
        
        logging.critical(f'Best news in the last minute: {bestnews} with a score of {bestscore}')
        logging.critical(f'Worst news in the last minute: {worstnews} with a score of {worstscore}')

        data = {'text': f'Best news in the last minute: {bestnews} with a score of {bestscore}'}
#        requests.post(url=webhook_url, json = data)
        data = {'text': f'Worst news in the last minute: {worstnews} with a score of {worstscore}'}
#        requests.post(url=webhook_url, json = data)

        oldminute = minutenow
        olddatetime = now.strftime('%Y-%m-%d %H:%M')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Slackbot: Reads tweets from postgres in regular intervals (time in minutes defined in config file, can be updated realtime)
Posts the best and worst news to the Slack channel (based on sentiment score)
"""

from sqlalchemy import create_engine
import time
from datetime import datetime, timedelta
import pandas as pd
import logging
import os
import requests

if __name__ == '__main__':
    time.sleep(1)
    pguser = os.environ.get('POSTGRES_USER')               # Import Postgres credentials form env file
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres') # pg connect
    webhook_url = os.environ.get('SLACK_WEBHOOK')           # Import Slack webhook from env file
    configdata=pd.read_csv('../config/config.csv')          # import settings: bot status (on/off) and the number of minutes between each post
    botswitch = configdata.iloc[0,0]
    minutes_to_post = int(configdata.iloc[0,2])
    timeframe_from = datetime.now().replace(microsecond=0)
    
    while True:
        time.sleep(1)       # check the following in every second:
        now = datetime.now().replace(microsecond=0)
        configdata=pd.read_csv('../config/config.csv')      # check for any realtime config chances (bot switch, post frequency)
        botswitch_new = configdata.iloc[0,0]
        minutes_to_post_new = int(configdata.iloc[0,2])
        if botswitch != botswitch_new:
            botswitch = botswitch_new
            logging.critical(f'\n\nSLACKBOT SWITCHED TO STATE: {botswitch}')    # log slackbot state switch
        if minutes_to_post != minutes_to_post_new:
            minutes_to_post = minutes_to_post_new
            logging.critical(f'\n\nTIME BETWEEN EACH SLACKBOT POST CHANGED TO: {minutes_to_post} MINUTE(S)')    # log post frequency time change
        if now > timeframe_from + timedelta(minutes = minutes_to_post):   # check if enough minutes have passed for the next slackbot posting
            if botswitch == 1:      # is the chatbot on?
                params = {'earliest': timeframe_from}                     # read out the new tweets from postgres
                out = pd.read_sql_query("SELECT * FROM tweets WHERE timestamp >= %(earliest)s", pg, params = params)
                if len(out) > 0:    # if there are new posts:
                    besttweet = out[out["sentimentscore"] == out["sentimentscore"].max()]   # check for tweet with highest sentiment score
                    bestnews = besttweet["tweet"].values[0]
                    bestscore = besttweet["sentimentscore"].values[0]
                    worsttweet = out[out["sentimentscore"] == out["sentimentscore"].min()]  # check for tweet with lowest sentiment score
                    worstnews = worsttweet["tweet"].values[0]
                    worstscore = worsttweet["sentimentscore"].values[0]        
                    logging.critical(f'Good news everyone! {bestnews} -- sentiment score: {bestscore}')   # logging...
                    logging.critical(f'Terrible news... {worstnews} -- sentiment score: {worstscore}')
                    logging.critical(f'Stay tuned for the next newsflash in {str(minutes_to_post)} minutes.')
                    data = {'text': f'Good news everyone! {bestnews}'}        # posting to Slack....
                    requests.post(url=webhook_url, json = data)
                    data = {'text': f'Terrible news... {worstnews}'}
                    requests.post(url=webhook_url, json = data)
                    data = {'text': f'Stay tuned for the next newsflash in {str(minutes_to_post)} minute(s).'}
                    requests.post(url=webhook_url, json = data)
            timeframe_from = datetime.now().replace(microsecond=0)  # set the "from" marker for the next posting cycle
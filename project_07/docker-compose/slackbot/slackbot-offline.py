#! python3
# slackbot.py - Slackbot script: Reads tweets from postgres, post them on Slack based on their sentiment score

import pandas as pd

configdata=pd.read_csv('./config/config.csv')
print(configdata.iloc[0,0])

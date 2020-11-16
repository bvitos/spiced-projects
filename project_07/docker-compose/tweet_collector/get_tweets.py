#import config
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
import json
import logging
import time
import pymongo
import os

def authenticate():
    """Function for handling Twitter Authentication. Credentials imported from env file
    """

    auth = OAuthHandler(os.environ.get('API_KEY'), os.environ.get('API_SECRET'))
    auth.set_access_token(os.environ.get('ACCESS_TOKEN'), os.environ.get('ACCESS_TOKEN_SECRET'))


    return auth

class TwitterListener(StreamListener):

    def on_data(self, data):

        """Whatever we put in this method defines what is done with
        every single tweet as it is intercepted in real-time"""

        t = json.loads(data) #t is just a regular python dictionary.

        tweet = {
        'text': t['text'],
        'username': t['user']['screen_name'],
        'followers_count': t['user']['followers_count']
        }

        tweetimport = tweet["text"] + " ... POSTED BY: " + tweet["username"]

        logging.critical(f'\n\n\nTWEET INCOMING: {tweetimport}\n\n\n')

        global collection
        collection.insert_one({"tweet": tweetimport})
        

#        logging.critical(f'\n\n\nTWEET INCOMING: {tweet["text"]}\n\n\n')

 #       global collection
 #       collection.insert_one({"tweet": tweet["text"]})

    def on_error(self, status):

        if status == 420:
            print(status)
            return False

if __name__ == '__main__':

    client = pymongo.MongoClient(host='mongodb', port=27017)
    db = client.test
    collection = db.tweets

    auth = authenticate()
    listener = TwitterListener()
    stream = Stream(auth, listener)
#    stream.filter(track=['bollywood', 'laxmii', 'akshaykumar', 'nasa', 'metaphysics', 'esoteric', 'tantric', 'sorcery', 'occult', ], languages=['en'])
    stream.filter(track=['nasa', 'astronomy', 'witchcraft', 'esoteria', 'esoteric', 'sorcery', 'occultism'], languages=['en'])
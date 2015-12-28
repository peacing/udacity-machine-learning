#Oscar Predictor
# This app will track tweets mentioning Oscar predictions and use machine learning to predict the category winners

import tweepy
import sys
#import sqlite3 as sql

# Connect to SQL database
#SQLdbFilePath = ''
#table_name = 'tweets'
#conn = sql.connect(SQLdbFilePath)

# User application credentials
#consumer_key = ''
#consumer_secret = ''

#access_token = ''
#access_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


class CustomStreamListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        super(tweepy.StreamListener, self).__init__()
        
    def on_status(self, status):
        print status.text , "\n"
        
        data = {}
        data['screen_name'] = status.user.screen_name    #Authors screen name/handle
        
        
        #c.execute("INSERT INTO {} VALUES (?,?,?,?,?,?,?,?,?,?)".format(table_name, (
        #    '@' + data['screen_name']
        #))
        
        #conn_commit()
        
sapi = tweepy.streaming.Stream(auth, CustomStreamListener(api))
sapi.silter(track=['criteria']



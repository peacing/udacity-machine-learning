#Oscar Predictor
# This app will track tweets mentioning Oscar predictions and use machine learning to predict the category winners

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
#import sys
#import sqlite3 as sql

# Connect to SQL database
#SQLdbFilePath = ''
#table_name = 'tweets'
#conn = sql.connect(SQLdbFilePath)

# User application credentials
consumer_key = 'nf1hSggwFmtWRC7CcKk0d6g6q'
consumer_secret = 'AYNPFeXzLTYHXtIvQaKUelHhz9Q0xmByPt53HWYdn2BmBIyC9X'

access_token = '271782849-8ESLV7V1UCbzAm1h8gqoPzryiEPt0PRgnhojhAqg'
access_secret = 'UjQ4pmBTOrgmHyU2LjHTK6cNRuddewtOnwq7pOoaieeqV'


class MyStreamListener(StreamListener):

    def on_data(self, data):
        print data
        return True
        
    def on_status(self, status):
        print status.text , "\n"
        
        data = {}
        data['screen_name'] = status.user.screen_name    #Authors screen name/handle
        
        
        #c.execute("INSERT INTO {} VALUES (?,?,?,?,?,?,?,?,?,?)".format(table_name, (
        #    '@' + data['screen_name']
        #))
        
        #conn_commit()
        
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False
            
if __name__ == '__main__':

    #handle twitter authentication and connection to stream api
    l = MyStreamListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    myStream = Stream(auth, l)   

    myStream.filter(track=['MLB'])



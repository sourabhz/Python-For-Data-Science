import tweepy
from textblob import TextBlob

# sentiment analysis is extracting and understanding human feelings from data

consumer_key = 'FkLunsUnUMSkCiiUDhWOOHXrU'
consumer_secret = '9GEQIXgIaKqsIgucs2B0Vx1Id5AzUt4dZU9IL4EhFl0bID5Jqn'

access_token = '2470523191-8f7FGmIoDzYLinNrn4liO7CtrQbswVaGksLSgQl'
access_token_secret = 'sc3jvSwQCyQRnpiCJW4STnKCEZvqx4yC9GkPMJyUu5YsW'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Narendra Modi')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)





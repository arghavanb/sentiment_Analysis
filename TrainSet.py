import pandas as pd
import numpy as np
import nltk
import sys
import json
import matplotlib.pyplot as plt
#import re
state_mapping = { 
	"alabama": "al",
	"alaska": "ak",
	"arizona": "az",
	"arkansas": "ar",
	"california": "ca",
	"colorado": "co",
	"connecticut": "ct",
	"delaware": "de",
	"florida": "fl",
	"georgia": "ga",
	"hawaii": "hi",
	"idaho": "id",
	"illinois": "il",
	"indiana": "in",
	"iowa": "ia",
	"kansas": "ks",
	"kentucky": "ky",
	"louisiana": "la",
	"maine": "me",
	"maryland": "md",
	"massachusetts": "ma",
	"michigan": "mi",
	"minnesota": "mn",
	"mississippi": "ms",
	"missouri": "mo",
	"montana": "mt",
	"nebraska": "ne",
	"nevada": "nv",
	"new hampshire": "nh",
	"new jersey": "nj",
	"new mexico": "nm",
	"new york": "ny",
	"north carolina": "nc",
	"north dakota": "nd",
	"ohio": "oh",
	"oklahoma": "ok",
	"oregon": "or",
	"pennsylvania": "pa",
	"rhode island": "ri",
	"south carolina": "sc",
	"south dakota": "sd",
	"tennessee": "tn",
	"texas": "tx",
	"utah": "ut",
	"vermont": "vt",
	"virginia": "va",
	"washington": "wa",
	"west virginia": "wv",
	"wisconsin": "wi",
	"wyoming": "wy" }
twitterData = 'output.txt'

""""read the labeled tweetts and extract features """
data = pd.read_csv("Sentiment Analysis Dataset1.csv")
#data = data.get_value
#conver dataframe to list of tuples
data = list(data.itertuples())
#print data[1][2]
print len(data)

tweets = []

for i in range(len(data)):
	sentiment = data[i][1]
	words = data[i][2]
	words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
	tweets.append((words_filtered, sentiment))

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
#design the Naive Bayesian classifier based on training sets
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
#read streeming tweets 

""" Read the streaming tweets about Hillary Clinton"""
def tweet_dict(twitterData):  
    ''' (file) -> list of dictionaries
    This method should take your output.txt
    file and create a list of dictionaries.
    '''
    twitter_list_dict = []
    twitterfile = open(twitterData)
    for line in twitterfile:
        twitter_list_dict.append(json.loads(line))
    return twitter_list_dict


TT = tweet_dict(twitterData)

print("number of tweets",len(TT))
State_Sentiment=[]
for t in TT:
    if 'text' in t.keys() and t['lang']=='en':
        tweet_words = t['text']
        tweet_state = t['user']['location']
        State_Sentiment.append((classifier.classify(extract_features(tweet_words.split())),tweet_state))
print State_Sentiment        

opinion=np.zeros(len(State_Sentiment))
state_opinion = []
for i in range(len(State_Sentiment )):
    opinion[i] = State_Sentiment[i][0]
    
    if not(State_Sentiment[i][1] ==None):
        
        loc = State_Sentiment[i][1].split()
        for l in loc:
            if l in state_mapping:
                state_opinion.append((state_mapping[l],opinion[i]))
                    
dic={}

for t in state_mapping.keys():

    dic[state_mapping[t]]=sum(x == state_mapping[t] for row in state_opinion for x in row)


#pie plot the sentiments 
labels = 'Positive Sentiment','Negative Sentiment'
sizes = [(opinion==1).sum(),(opinion==0).sum()]
colors = ['lightskyblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
fig = plt.figure()
ax = fig.gca()    
plt.show()
#barchart plot for rate of negative sentiment in each state
import pylab as pl
import numpy as np

X = np.arange(len(dic))
pl.bar(X, np.array(dic.values())/float(sum(dic.values())), align='center', width=0.5)
pl.xticks(X, dic.keys())
pl.figure(figsize=(10, 10)) 
pl.ylim(0, .2)
pl.show()
import pickle
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import re
from flask import Flask,jsonify,json,request
from flask_restful import reqparse

f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

teks = open('data_latih_hasil_praproses.json', 'r')
clean_tweet= json.loads(teks)

tweets=[]
for word,sentimen in clean_tweet:
    words_filtered = [e.lower() for e in word.split()]
    tweets.append((words_filtered, sentimen))


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

i = "RT @JktMajuBersama: Pesan dari Mas @aniesbaswedan untuk teman-teman mahasiswa. \n\n#SalamBersama #MajuBersama #CoblosNomor3 https://t.co/y4hK\u2026"
sentimen= classifier.classify(extract_features(i.split()))

print(sentimen)
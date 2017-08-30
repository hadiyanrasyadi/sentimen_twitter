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
import json


f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

teks = open('clean_tweet.json', 'r')
teks_open = teks.read()
clean_tweet= json.loads(teks_open)
teks.close()

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

app = Flask(__name__)
@app.route('/predict', methods=["POST"])

def predict():
	try:
	
		# Parse the needed arguments on POST request
		parser = reqparse.RequestParser()
		parser.add_argument('idalat', type=str, help='idalatt')
		args = parser.parse_args()
		teks = args['idalat']

		i=teks;
		sentimen= classifier.classify(extract_features(i.split()))

		return jsonify({"tasks" : sentimen})

	except Exception as e:
		return {'error': str(e)}
	
	
	#return "hello world"+idalat
if __name__=="__main__":
    app.run(host= '0.0.0.0', port=33, debug=True)

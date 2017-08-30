import pickle
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import re

f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
	    features['contains(%s)' % word] = (word in document_words)
	return features

i = "RT @JktMajuBersama: Pesan dari Mas @aniesbaswedan untuk teman-teman mahasiswa. \n\n#SalamBersama #MajuBersama #CoblosNomor3 https://t.co/y4hK\u2026"
sentimen= classifier.classify(extract_features(i.split()))

print(sentimen)
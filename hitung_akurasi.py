import string
import json
import sys
import csv

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import ELEProbDist, FreqDist
from nltk import NaiveBayesClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import re

#untuk membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def main():
	fileLatih = 'train'
	fileUji = 'test'
	with open(fileLatih+".json",'r') as f, open('key_norm.csv') as filecsv, open('stopword_list_TALA.txt','r') as stp_file, open(fileUji+"_tesakurasi_praproses.txt",'w') as hasil_tweet, open(fileUji+".json",'r') as fOpen, open(fileUji+"_testakurasi_hasil.txt",'w') as hasil_akurasi:


		json_str = f.read()
		json_data = json.loads(json_str)

		json_strOpen=fOpen.read()
		json_dataOpen=json.loads(json_strOpen)

		# ini mengubah karakter unik u' dalam delimiter nya dan mengambil value dari isi
		utfjson = []
		for js in json_data:
			temp = dict()
			for key,val in js.iteritems():
				if key.encode("utf-8") == "isi": 
					temp[key.encode("utf-8")] = val.encode("utf-8")
			utfjson.append(temp)

		#mengambil value dari dictionary 'isi'
		isi_dict=[]
		for reg in utfjson:
			for k,v in reg.iteritems():
				regular =  v.lower()
				isi_dict.append(regular)
		
		print('Membersihkan karakter2 khusus seperti link web, akun suser')
		# Membersihkan karakter2 khusus seperti link web, akun suser, dan #
		clean_regular = []
		for regular in isi_dict:
			regular = re.sub('((www\.[\s]+)|(https?://[^\s]+))',' ',regular)
			regular = re.sub('@[^\s]+',' ',regular)
			regular = re.sub('[\s]+', ' ', regular)
			regular = re.sub('#([^\s]+)', ' ', regular)
			regular = regular.strip('\"')
			clean_regular.append(regular)

		# melakukan normalisasi teks, mengubah singkatan2 yang ada
		read = csv.reader(filecsv, delimiter=',')
		keys=[]
		results=[]
		clean_norm=[]
		
		for row in read: # untuk mengubah csv ke dalam array
			keys.append(row[1])
			results.append(row[2])
		
		gabung=dict(zip(keys,results)) #menggabungkan dua array

		print('Mulai normalisasi')
		for q in clean_regular: 
			temp = q.split()
			# Ambil array Kata
			for i,_ in enumerate(temp):
				if temp[i] in gabung:
					temp[i] = gabung[temp[i]]
			temp2=' '.join(temp)
			clean_norm.append(temp2)

		print('mulai stemming..')
		# Proses stemming data
		clean_stemmer = []
		for csm in clean_norm:
			clean = stemmer.stem(csm)
			clean_stemmer.append(clean)

		# Membersihkan Stopword
		atp = [] #variabel menyimpan array dalam array
		stp=[] #variabel mengubah atp menjadi 1 array saja
		clean_stopword = []
		#mengubah stopword yang txt ke bentuk array
		for line in stp_file:
		    atp.append(line.strip().split('/n'))

		stp=sum(atp,[])
		print ('Clean Stopword')
		for csw in clean_stemmer:
			temp=csw.split() #membuat tokenize
			clean_pnc = filter(lambda x: x not in string.punctuation,temp)
			clean_sw = filter(lambda x: x not in stp,clean_pnc)
			cc=' '.join(clean_sw) #menggabngkan tokenize
			clean_stopword.append(cc)

		# untuk mengambbil sentimen aja
		utfjson3 = [] 
		for js in json_data:
			temp = dict()
			for key,val in js.iteritems():
				if key.encode("utf-8") == "sentimen": 
					temp[key.encode("utf-8")] = val.encode("utf-8")
			utfjson3.append(temp)

		b=[] #variabel untuk menyiman kumulan sentimen
		for reg in utfjson3:
			for k,v in reg.iteritems():
				regular =  v.lower()
				b.append(regular)

		clean_tweet=[]
		for i in range(len(clean_stopword)):
			c = (clean_stopword[i],b[i])
			clean_tweet.append(c)

		print ('menulis ke file '+fileUji)
		#menulis hasil tweet
		hasil_tweet.write('[\n')
		FNL= False
		for i in clean_tweet:
			if FNL == True:
				hasil_tweet.write(',\n')
			FNL = True
			hasil_tweet.write(str(i))

		hasil_tweet.write(']')

		print('Donee ^^')

		############ Membuat (tokenize word, sentimen) ###############
		#untuk menyimpan array kata yang sudah di split 
		print('Proses Sentimen')
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


		#untuk ekstraksi fitur
		def extract_features(document):
		    document_words = set(document)
		    features = {}
		    for word in word_features:
		        features['contains(%s)' % word] = (word in document_words)
		    return features


		training_set = nltk.classify.apply_features(extract_features, tweets)


		##########################  Membuat train klasifier  #################################
		classifier = nltk.NaiveBayesClassifier.train(training_set)

		def train(labeled_featuresets, estimator=ELEProbDist):
		    # Create the P(label) distribution
		    label_probdist = estimator(label_freqdist)
		    # Create the P(fval|label, fname) distribution
		    feature_probdist = {}
		    return NaiveBayesClassifier(label_probdist, feature_probdist)

		########################################
		## Membuka file yang akan di sentimen ##
		########################################
		utfjson2_open=[]
		for jso in json_dataOpen:
			temp = dict()
			for key,val in jso.iteritems():
				if key.encode("utf-8") == "sentimen": 
					temp[key.encode("utf-8")] = val.encode("utf-8")
			utfjson2_open.append(temp)

		# mengambil isi yang akan di sentimen
		isi2_open=[]
		for rego in utfjson2_open:
			for k,v in rego.iteritems():
				regularo =  v.lower()
				isi2_open.append(regularo)

		#melakukan sentimen
		print('Melakukan sentimen')
		hasil_sentimen=[]
		for i in isi2_open:
			sentimen= classifier.classify(extract_features(i.split()))
			hasil_sentimen.append(sentimen)

		################## Hitung Akurasi ###########
		positif = 0
		positif_tp=0
		positif_fn=0
		negatif = 0
		negatif_tp=0
		negatif_fn=0
		netral = 0
		netral_tp=0
		netral_fn=0

		for i in range(len(isi2_open)):
			if isi2_open[i] == "netral":
				if isi2_open[i] == hasil_sentimen[i]:
					netral_tp = netral_tp + 1
					continue
				else:
					netral_fn = netral_fn + 1
					continue
			elif isi2_open[i] == "positif":
				if isi2_open[i] == hasil_sentimen[i]:
					positif_tp = positif_tp + 1
					continue
				else:
					positif_fn = positif_fn + 1
					continue
			else:
				if isi2_open[i] == hasil_sentimen[i]:
					negatif_tp = negatif_tp + 1
					continue
				else:
					negatif_fn = negatif_fn + 1
					continue


		for i in range(len(isi2_open)):
			if isi2_open[i] == "netral":
				netral = netral + 1
				continue
			elif isi2_open[i] == "positif":
				positif = positif + 1
				continue
			else:
				negatif = negatif + 1
				continue

		akurasi = (float(positif_tp+negatif_tp+netral_tp)) / (positif+negatif+netral) * 100
		
		hasil_akurasi.write('Berikut Hasil sentimen dengan akurasi : '+ str(akurasi) +' %\n')
		hasil_akurasi.write ('positif : '+ str(positif) + '\n')
		hasil_akurasi.write ('positif_tp : ' + str(positif_tp) + '\n')
		hasil_akurasi.write ('positif_fn : ' + str(positif_fn) + '\n\n')

		hasil_akurasi.write ('negatif : '+ str(negatif) + '\n')
		hasil_akurasi.write ('negatif_tp : ' + str(negatif_tp) + '\n')
		hasil_akurasi.write ('negatif_fn : ' + str(negatif_fn) + '\n\n')

		hasil_akurasi.write ('netral : '+ str(netral) + '\n')
		hasil_akurasi.write ('netral_tp : ' + str(netral_tp) + '\n')
		hasil_akurasi.write ('netral_fn : ' + str(netral_fn) + '\n\n')

		hasil_tweet.close()
		hasil_akurasi.close()

if __name__ == '__main__':
	main()
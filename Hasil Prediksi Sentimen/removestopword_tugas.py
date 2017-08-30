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
	fileLatih = 'data_latih'
	fileUji = sys.argv[1]
	with open(fileLatih+".json",'r') as f, open(fileLatih+'_Hasil_normalisasi.txt','w') as hasil_norm, open(fileLatih+'_Hasil_stemming.txt','w') as hasil_stem, open(fileLatih+"_hasil_stopword.txt",'w') as hasil_stop, open(fileLatih+"_hasil_praproses.txt",'w') as hasil_tweet, open('key_norm.csv') as filecsv, open('stopword_list_TALA.txt','r') as stp_file, open(fileUji+".json",'r') as fOpen, open(fileUji+"_hasil_sentimen.txt",'w') as hasil_sentimen:

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
		clean_digit=[]
		for s in clean_regular:
			c_num = ''.join(i for i in s if not i.isdigit())
			clean_digit.append(c_num)


		for q in clean_digit: 
			temp = q.split()
			# Ambil array Kata
			for i,_ in enumerate(temp):
				if temp[i] in gabung:
					temp[i] = gabung[temp[i]]
			temp2=' '.join(temp)
			clean_norm.append(temp2)

		##### Menulis hasil Normalisasi ########
		print ('menulis hasil normalisasi')
		#menulis hasil tweet
		hasil_norm.write('[\n')
		FNL= False
		for i in clean_norm:
			if FNL == True:
				hasil_norm.write(',\n')
			FNL = True
			hasil_norm.write(str(i))

		hasil_norm.write(']')

		print('Donee menulis hasil normalisasi ^^')
		#########################################

		print('mulai stemming..')
		# Proses stemming data
		clean_stemmer = []
		for csm in clean_norm:
			clean = stemmer.stem(csm)
			clean_stemmer.append(clean)

		##### Menulis hasil Normalisasi ########
		print ('menulis hasil Stemming')
		#menulis hasil tweet
		hasil_stem.write('[\n')
		FNL= False
		for i in clean_stemmer:
			if FNL == True:
				hasil_stem.write(',\n')
			FNL = True
			hasil_stem.write(str(i))

		hasil_stem.write(']')

		print('Donee menulis hasil Stemming ^^')
		#########################################


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

		##### Menulis hasil Normalisasi ########
		print ('menulis hasil stopword removal')
		#menulis hasil tweet
		hasil_stop.write('[\n')
		FNL= False
		for i in clean_stopword:
			if FNL == True:
				hasil_stop.write(',\n')
			FNL = True
			hasil_stop.write(str(i))

		hasil_stop.write(']')

		print('Donee menulis hasil stopword removal ^^')
		#########################################

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

		print ('menulis ke file '+fileLatih)
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
		utfjson_open=[]
		for jso in json_dataOpen:
			temp = dict()
			for key,val in jso.iteritems():
				if key.encode("utf-8") == "isi": 
					temp[key.encode("utf-8")] = val.encode("utf-8")
			utfjson_open.append(temp)

		# mengambil isi yang akan di sentimen
		isi_open=[]
		for rego in utfjson_open:
			for k,v in rego.iteritems():
				regularo =  v.lower()
				isi_open.append(regularo)

		#melakukan sentimen
		print('Melakukan sentimen')
		result_sentimen=[]
		for i in isi_open:
			sentimen= classifier.classify(extract_features(i.split()))
			result_sentimen.append(sentimen)

		print ('Menghitung hasil sentimen')
		positif=result_sentimen.count("positif")
		negatif=result_sentimen.count("negatif")
		netral=result_sentimen.count("netral")
		
		total=positif+negatif+netral
		
		presen_positif=(float(positif)/total)*100
		presen_negatif=(float(negatif)/total)*100
		presen_netral=(float(netral)/total)*100
		
		print 'Presentase Positif: '
		print positif
		print '---'
		print 'Presentase Negatif: '
		print negatif
		print '---'
		print 'Presentase Netral: '
		print netral
		print '---'

		
		hasil_sentimen.write('Berikut hasil sentimen analisis dari akun '+fileUji+ ' sebanyak ' + str(total) + 'data.\n')
		hasil_sentimen.write('Jumlah tweet dan retweet bersentimen positif : '+str(positif)+'\n')
		hasil_sentimen.write('Jumlah tweet dan retweet bersentimen negatif : '+str(negatif)+'\n')
		hasil_sentimen.write('Jumlah tweet dan retweet bersentimen netral : '+str(netral)+'\n\n')

		hasil_sentimen.write('Dalam bentuk presentase :\n')
		hasil_sentimen.write('Persentase tweet dan retweet bersentimen positif : '+str(presen_positif)+'%\n')
		hasil_sentimen.write('Persentase tweet dan retweet bersentimen negatif : '+str(presen_negatif)+'%\n')
		hasil_sentimen.write('Persentase tweet dan retweet bersentimen netral : '+str(presen_netral)+'%\n\n')
		hasil_sentimen.close()
		hasil_tweet.close()
		

if __name__ == '__main__':
	main()
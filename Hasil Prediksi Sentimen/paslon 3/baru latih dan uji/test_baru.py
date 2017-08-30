import string
import json
import sys
import csv
import pickle

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
	fileLatih = 'data_latih2'
	fileUji = sys.argv[1]
	with open(fileLatih+".json",'r') as f, open('Hasil Informative Feature.txt', 'w') as file, open(fileLatih+'_hasil_normalisasi.txt','w') as hasil_norm, open(fileLatih+'_hasil_stemming.txt','w') as hasil_stem, open(fileLatih+"_hasil_stopword.txt",'w') as hasil_stop, open(fileLatih+"_hasil_praproses.txt",'w') as hasil_tweet,open('key_norm.csv') as filecsv, open('stopword_list_TALA.txt','r') as stp_file, open(fileUji+".json",'r') as fOpen, open(fileUji+"_hasil_akurasi.txt",'w') as hasil_akurasi, open('my_classifier.pickle', 'wb') as model:

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
			hasil_norm.write('"('+str(i)+')"')

		hasil_norm.write(']')

		print('Donee menulis hasil normalisasi ^^')
		#########################################

			
		print('mulai stemming..')
		# Proses stemming data
		clean_stemmer = []
		for csm in clean_norm:
			clean = stemmer.stem(csm)
			clean_stemmer.append(clean)

		##### Menulis hasil Stemming ########
		print ('menulis hasil Stemming')
		#menulis hasil tweet
		hasil_stem.write('[\n')
		FNL= False
		for i in clean_stemmer:
			if FNL == True:
				hasil_stem.write(',\n')
			FNL = True
			hasil_stem.write('"('+str(i)+')"')

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
			hasil_stop.write('"('+str(i)+')"')

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

		print ('menulis ke file '+fileUji)
		#menulis hasil tweet
		hasil_tweet.write('[\n')
		FNL= False
		for i in clean_tweet:
			if FNL == True:
				hasil_tweet.write(',\n')
			FNL = True
			hasil_tweet.write('"'+str(i)+'"')

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

		pickle.dump(classifier, model)
		########################################
		## Membuka file yang akan di sentimen ##
		########################################
		utfjson2_open=[]
		for jso in json_dataOpen:
			temp = dict()
			for key,val in jso.iteritems():
				if key.encode("utf-8") == "isi": 
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

		########################################
		## Ambil sentimen awal banget         ##
		########################################
		utfjson2_sentimen=[]
		for jso in json_dataOpen:
			temp = dict()
			for key,val in jso.iteritems():
				if key.encode("utf-8") == "sentimen": 
					temp[key.encode("utf-8")] = val.encode("utf-8")
			utfjson2_sentimen.append(temp)

		# mengambil isi yang akan di sentimen
		isi2_sentimen=[]
		for rego in utfjson2_sentimen:
			for k,v in rego.iteritems():
				regularo =  v.lower()
				isi2_sentimen.append(regularo)

		def show_most_informative_features(self, n=10):
		    strlist = []
		    # Determine the most relevant features, and display them.
		    cpdist = self._feature_probdist
		    # print('Most Informative Features')
		    strlist.append('Most Informative Features')
		    for (fname, fval) in self.most_informative_features(n):
		            def labelprob(l):
		                return cpdist[l,fname].prob(fval)
		            labels = sorted([l for l in self._labels
		                     if fval in cpdist[l,fname].samples()],
		                    key=labelprob)
		            if len(labels) == 1: continue
		            l0 = labels[0]
		            l1 = labels[-1]
		            if cpdist[l0,fname].prob(fval) == 0:
		                ratio = 'INF'
		            else:
		                ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) /
		                          cpdist[l0,fname].prob(fval))
		            # print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
		            #      (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))
		            strlist.append(('%24s = %-14r %6s : %-6s = %s : 1.0' %
		                          (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))
		    return strlist

		list = show_most_informative_features(classifier, 500)
		for i in list:
			print>>file,i




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

		for i in range(len(isi2_sentimen)):
			if isi2_sentimen[i] == "netral":
				if isi2_sentimen[i] == hasil_sentimen[i]:
					netral_tp = netral_tp + 1
					continue
				else:
					netral_fn = netral_fn + 1
					continue
			elif isi2_sentimen[i] == "positif":
				if isi2_sentimen[i] == hasil_sentimen[i]:
					positif_tp = positif_tp + 1
					continue
				else:
					positif_fn = positif_fn + 1
					continue
			else:
				if isi2_sentimen[i] == hasil_sentimen[i]:
					negatif_tp = negatif_tp + 1
					continue
				else:
					negatif_fn = negatif_fn + 1
					continue
################################################################
		pos_pos=0
		pos_neg=0
		pos_net=0

		neg_neg=0
		neg_pos=0
		neg_net=0

		net_net=0
		net_pos=0
		net_neg=0


		for i in range(len(isi2_sentimen)):
			if isi2_sentimen[i] == "netral":
				if isi2_sentimen[i] == hasil_sentimen[i]:
					net_net = net_net + 1
					continue
				elif hasil_sentimen[i] == "positif":
					net_pos = net_pos + 1
					continue
				else:
					net_neg = net_neg + 1
			elif isi2_sentimen[i] == "positif":
				if isi2_sentimen[i] == hasil_sentimen[i]:
					pos_pos = pos_pos + 1
					continue
				elif hasil_sentimen[i] == "negatif":
					pos_neg = pos_neg + 1
					continue
				else:
					pos_net = pos_net + 1
			else:
				if isi2_sentimen[i] == hasil_sentimen[i]:
					neg_neg = neg_neg + 1
					continue
				elif hasil_sentimen[i] == "positif":
					neg_pos = neg_pos + 1
					continue
				else:
					neg_net = neg_net + 1


		for i in range(len(isi2_sentimen)):
			if isi2_sentimen[i] == "netral":
				netral = netral + 1
				continue
			elif isi2_sentimen[i] == "positif":
				positif = positif + 1
				continue
			else:
				negatif = negatif + 1
				continue

		akurasi = (float(positif_tp+negatif_tp+netral_tp)) / (positif+negatif+netral) * 100
		#Sensitivity 
		positif_sensitivity=float(positif_tp)/(positif_tp+positif_fn)
		negatif_sensitivity=float(negatif_tp)/(negatif_tp+negatif_fn)
		netral_sensitivity=float(netral_tp)/(netral_tp+netral_fn)

		#Spesitivity
		positif_spesivisity=float(neg_neg+neg_net+net_neg+net_net)/(neg_neg+neg_net+net_neg+net_net+neg_pos+net_pos)
		negatif_spesivisity=float(pos_pos+pos_net+net_pos+net_net)/(pos_pos+pos_net+net_pos+net_net+pos_neg+net_neg)
		netral_spesivisity=float(pos_pos+pos_neg+neg_pos+neg_neg)/(pos_pos+pos_neg+neg_pos+neg_neg+pos_net+neg_net)

		hasil_akurasi.write('{\n')
		hasil_akurasi.write('"akurasi_sistem" : '+ str(akurasi) +',\n')

		hasil_akurasi.write ('"positif" : '+ str(positif) + ',\n')
		hasil_akurasi.write ('"positif_tp" : ' + str(positif_tp) + ',\n')
		hasil_akurasi.write ('"positif_fn" : ' + str(positif_fn) + ',\n')
		hasil_akurasi.write ('"positif_sensitivity" : ' + str(positif_sensitivity) + ',\n')
		hasil_akurasi.write ('"positif_spesivisity" : ' + str(positif_spesivisity) + ',\n\n')
		

		hasil_akurasi.write ('"negatif" : '+ str(negatif) + ',\n')
		hasil_akurasi.write ('"negatif_tp" : ' + str(negatif_tp) + ',\n')
		hasil_akurasi.write ('"negatif_fn" : ' + str(negatif_fn) + ',\n')
		hasil_akurasi.write ('"negatif_sensitivity" : ' + str(negatif_sensitivity) + ',\n')
		hasil_akurasi.write ('"negatif_spesivisity" : ' + str(negatif_spesivisity) + ',\n\n')
		

		hasil_akurasi.write ('"netral" : '+ str(netral) + ',\n')
		hasil_akurasi.write ('"netral_tp" : ' + str(netral_tp) + ',\n')
		hasil_akurasi.write ('"netral_fn" : ' + str(netral_fn) + ',\n')
		hasil_akurasi.write ('"netral_sensitivity" : ' + str(netral_sensitivity) + ',\n')
		hasil_akurasi.write ('"netral_spesivisity" : ' + str(netral_spesivisity) + ',\n\n')
		
		hasil_akurasi.write ('"Aktual positif, prediksi positif" : ' + str(pos_pos) + ',\n')
		hasil_akurasi.write ('"Aktual positif, prediksi negatif" : ' + str(pos_neg) + ',\n')
		hasil_akurasi.write ('"Aktual positif, prediksi netral" : ' + str(pos_net) + ',\n\n')
		hasil_akurasi.write ('"Aktual negatif, prediksi negatif" : ' + str(neg_neg) + ',\n')
		hasil_akurasi.write ('"Aktual negatif, prediksi positif" : ' + str(neg_pos) + ',\n')
		hasil_akurasi.write ('"Aktual negatif, prediksi netral" : ' + str(neg_net) + ',\n\n')
		hasil_akurasi.write ('"Aktual netral, prediksi netral" : ' + str(neg_net) + ',\n')
		hasil_akurasi.write ('"Aktual netral, prediksi positif" : ' + str(neg_pos) + ',\n')
		hasil_akurasi.write ('"Aktual netral, prediksi negatif" : ' + str(neg_neg) + '\n')
		hasil_akurasi.write ('}')


		hasil_tweet.close()
		hasil_akurasi.close()
		file.close()

if __name__ == '__main__':
	main()
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
	fileLatih = 'data_latih'
	fileUji = sys.argv[1]
	with open(fileLatih+".json",'r') as f, open('Hasil Informative Feature.txt', 'w') as file, open(fileLatih+'_hasil_normalisasi.txt','w') as hasil_norm, open(fileLatih+'_hasil_stemming.txt','w') as hasil_stem, open(fileLatih+"_hasil_stopword.txt",'w') as hasil_stop, open(fileLatih+"_hasil_praproses.txt",'w') as hasil_tweet,open('key_norm.csv') as filecsv, open('stopword_list_TALA.txt','r') as stp_file, open(fileUji+".json",'r') as fOpen, open(fileUji+"_hasil_akurasi.txt",'w') as hasil_akurasi, open('my_classifier.pickle', 'wb') as model, open('clean_tweet.json', 'w') as decode:

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

		json.dump(clean_tweet,decode)
		
if __name__ == '__main__':
	main()
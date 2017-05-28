from nltk.tokenize import word_tokenize
import string
import json
import re
import csv
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

FileResult="Tweets_PreProcessed_"+fileName+".txt"

#untuk membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def main():
	fileName = sys.argv[1]

	with open(fileName+".json",'r') as f, open('key_norm.csv') as filecsv, open('StopWords_Eng-Ind.txt','r') as stp_file:
		json_str = f.read()
		json_data = json.loads(json_str)

		# ini mengubah karakter unik u' dalam delimiter nya
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

		for q in clean_regular: #proses pengecekan, apakah ada kata yang harus dinormalisasi
			temp = q
			for k,v in gabung.iteritems():
				temp=temp.replace(k,v)
			clean_norm.append(temp)

		# Membersihkan Stopword
		atp = [] #variabel menyimpan array dalam array
		stp=[] #variabel mengubah atp menjadi 1 array saja
		clean_stopword = []
		#mengubah stopword yang txt ke bentuk array
		for line in stp_file:
		    atp.append(line.strip().split('/n'))
		stp=sum(atp,[])

		for csw in clean_norm:
			temp=csw.split() #membuat tokenize
			clean_pnc = filter(lambda x: x not in string.punctuation,temp)
			clean_sw = filter(lambda x: x not in stp,clean_pnc)
			cc=' '.join(clean_sw) #menggabngkan tokenize
			clean_stopword.append(cc)

		# Proses stemming data
		clean_stemmer = []
		for csm in clean_stopword:
			clean = stemmer.stem(csm)
			clean_stemmer.append(clean)

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
		for i in range(len(clean_stemmer)):
			c = (clean_stemmer[i],b[i])
			clean_tweet.append(c)

if __name__ == '__main__':
	main()
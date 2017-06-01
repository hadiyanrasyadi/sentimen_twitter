from nltk.tokenize import word_tokenize
import string
import json
import re
import sys
import csv
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#untuk membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def main():
	fileName = sys.argv[1]

	with open(fileName+".json",'r') as f, open('key_norm.csv') as filecsv, open('stopword_list_TALA.txt','r') as stp_file, open(fileName+".txt",'w') as hasil_tweet:
		json_str = f.read()
		json_data = json.loads(json_str)

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

		print ('menulis ke file '+fileName)
		#menulis hasil tweet
		hasil_tweet.write('[\n')
		FNL= False
		for i in clean_tweet:
			if FNL == True:
				hasil_tweet.write(',\n')
			FNL = True
			hasil_tweet.write(str(i))

		hasil_tweet.write(']')
		hasil_tweet.close()
		print('Donee ^^')
if __name__ == '__main__':
	main()
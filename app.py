import math

from flask import Flask, render_template, request, flash
from random import shuffle
import pandas as pd

app = Flask(__name__)
app.secret_key = 'super secret'

@app.route('/', methods=['GET','POST'])
def index():

	if request.method == 'POST':

		lama_usaha		= request.form['lama_usaha']
		jumlah_pekerja	= request.form['jumlah_pekerja']
		omzet 			= request.form['omzet']
		jumlah_aset		= request.form['jumlah_aset']
		k 				= request.form['k']

		if not lama_usaha:
			flash('Telah berapa lama usaha anda berjalan ?','warning')
			return render_template('base.html')
		if not jumlah_pekerja:
			flash('Berapa jumlah pekerja yang anda miliki ?','warning')
			return render_template('base.html')
		if not omzet:
			flash('Berapa omzet yang anda dapatkan ?','warning')
			return render_template('base.html')
		if not jumlah_aset:
			flash('Berapa jumlah aset yang anda miliki ?','warning')
			return render_template('base.html')
		if not k:
			flash('Silahkan tentukan jarak ke tetanggaan','warning')
			return render_template('base.html')

		items = ReadData('dataset.txt')

		lama_usaha		= int(lama_usaha)
		jumlah_pekerja	= int(jumlah_pekerja)
		omzet 			= int(omzet)
		jumlah_aset		= int(jumlah_aset)
		k 				= int(k)
		
		#newItem = {'LamaUsaha' : 4, 'Pekerja' : 15, 'Omzet' : 4, 'JumlahAset' : 6 }
		newItem = {'LamaUsaha' : lama_usaha, 'Pekerja' : jumlah_pekerja, 'Omzet' : omzet, 'JumlahAset' : jumlah_aset }

		accuracy2, maxi, count, neighbors = Classify(newItem, k, items)

		skor = pd.DataFrame(neighbors, columns = ['Skor' , 'Hasil Keputusan'])
		keputusan = pd.DataFrame(list(count.items()), columns=['Hasil Keputusan', 'Jumlah'])

		return render_template('base.html', keputusan=keputusan.to_html(classes='table table-striped table-hover table-bordered table-sm table-responsive-sm'), skor=skor.to_html(classes='table table-striped table-hover table-bordered table-sm table-responsive-sm'), lama_usaha=lama_usaha, jumlah_pekerja=jumlah_pekerja, omzet=omzet, jumlah_aset=jumlah_aset, evaluat=accuracy2, items=items, k=k)
	return render_template('base.html')

@app.route('/dataset')
def dataset():
	items = ReadData('dataset.txt')
	items = pd.DataFrame(items)
	return render_template('dataset.html',items=items.to_html(classes='table table-striped table-hover table-bordered table-sm table-responsive-sm'))


def readDataTest(fileName):
	#dataset
	f = open(fileName, 'r')
	lines = f.read().splitlines()
	f.close()

	#feature
	features = lines[0].split(',')[:-1]

	#list
	items = []
	for i in range(1, len(lines)):
		line = lines[i].split(',')
		itemFeatures = {'Class': line[-1]}
		#Iterate feature
		for j in range(len(features)):
			# Get the feature at index j
			f = features[j]
			# Convert feature value to float
			v = float(line[j])
			# Add feature value to dict
			itemFeatures[f] = v
		#Append temp dict to items
		items.append(itemFeatures)
	shuffle(items) 
	return items

def ReadData(fileName):
	#dataset
	f = open(fileName, 'r')
	lines = f.read().splitlines()
	f.close()

	#feature
	features = lines[0].split(',')[:-1]

	#list
	items = []
	for i in range(1, len(lines)):
		line = lines[i].split(',')
		itemFeatures = {'Class': line[-1]}
		#Iterate feature
		for j in range(len(features)):
			# Get the feature at index j
			f = features[j]
			# Convert feature value to float
			v = float(line[j])
			# Add feature value to dict
			itemFeatures[f] = v
		#Append temp dict to items
		items.append(itemFeatures)
	#shuffle(items) 
	return items

def EuclideanDistance(x,y):
	S=0
	for key in x.keys():
		S += math.pow(x[key]-y[key], 2)
	return math.sqrt(S)

def CalculateNeighborsClass(neighbors,k):
	count = {}
	for i in range(k):
		if neighbors[i][1] not in count:
			count[neighbors[i][1]] = 1
		else:
			count[neighbors[i][1]] += 1
	return count

def FindMax(Dict):
	maximum = -1
	classification = ''

	for key in Dict.keys():
		if Dict[key] > maximum:
			maximum = Dict[key]
			classification = key
	return classification, maximum

def Classify(nItem, k, Items):
	if (k > len(Items)):
		return "k larger than list length"

	neighbors = []
	distance2 =  []
	for item in Items:
		#Find Euclidean Distance
		distance = EuclideanDistance(nItem, item)
		distance2.append(distance)
		#Update neigbors
		neigbors = UpdateNeighbors(neighbors,item,distance,k)
	#Count number each class
	count = CalculateNeighborsClass(neighbors, k)
	#find the max in count / class with the most appearances
	klas,maxi = FindMax(count)

	return klas, maxi, count, neighbors

def UpdateNeighbors(neighbors,item,distance,k):
	if len(neighbors) < k:
		neighbors.append([distance, item['Class']])
		neighbors = sorted(neighbors)
	else:
		if neighbors[-1][0] > distance:
			neighbors[-1] = [distance, item['Class']]
			neighbors = sorted(neighbors)
	return neighbors

#Fungsi Evaluasi
def K_FoldValidation(K,k,Items):

	hasil = [];
	maxi = [];
	if K > len(Items):
		return -1
	correct = 1
	total = len(Items)*(K-1)
	l = int(len(Items)/K)

	for i in range(K):
		#split dataset into training and testing
		trainingSet = Items[i * l:(i + 1) * l]
		item2 = readDataTest('test.txt')

		newItem = {'LamaUsaha' : 4, 'Pekerja' : 15, 'Omzet' : 4, 'JumlahAset' : 6}
		testSet = item2[:i * l] + item2[(i + 1) * l:]

		for item in testSet:
			itemClass = item['Class']
			itemFeatures = {}

			for key in item:
				if key != 'Class':
					itemFeatures[key]=item[key]
			guess,maxi = Classify(newItem,k,trainingSet)
			if guess == itemClass:
				correct +=1

			hasil.append(guess)
			maxi.append(maxi)
	#accuracy = correct / float(total)
	return hasil, maxi

def Evaluate(K,k,items, iterations):

	accuracy = 0
	for i in range(iterations):
		shuffle(items)
		accuracy2,hasil = K_FoldValidation(K,k,items)
		accuracy += accuracy2

	accuracy = accuracy/float(iterations)

	return accuracy,yhasil

if __name__ == '__main__':
	app.run(debug=True, port=5000)
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
#import gensim.downloader as gensim_downloader
import numpy as np
import gensim

import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
#automatically detect common phrases (bigrams) from a list of sentences.
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import os
import random
import math
import string
from Utils import Utils
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

class DataCleaning(object):
	#Testing data embedding is also stored in here, as we are using same model for traing
	#Lambda Functions - Start
	def cleanString(self, text: str, ifOutlireCharRemoveNeeded = False)-> str:
		#remove punctuation
		text = "".join([char for char in text if char not in string.punctuation])
		if ifOutlireCharRemoveNeeded:
			text = "".join([char for char in text if char not in ['N', 'K', 'M', 'R', 'S', 'W', 'Y']])
		# Convert all to lower
			#not needed in here
		# Remove Stop Words
			#not needed in here
		return text

	def appendCounterDict(self, wordDict: dict)-> None:
		for k in wordDict:
			self.wordCounter[k]+=wordDict[k]
		return

	def splitWords(self, dna_seq: str, word_len: int)-> str:
		strArray = [dna_seq[index : index + word_len] for index in range(0, len(dna_seq), word_len)]
		self.totalUniqueWords.update(strArray)
		self.appendCounterDict(dict(Counter(strArray)))
		text = " ".join(strArray)
		self.maxWordLen = max(self.maxWordLen, len(strArray))
		return text
	#Lambda Functions - End

	def __init__(self, train_x_file = "train_features.csv", train_y_file = "train_labels.csv", test_x_file = "test_features.csv"):
		dataX = pd.read_csv(Utils.getAbsFilePath(train_x_file), index_col=0)
		self.maxWordLen = 0
		self.total_data = dataX
		dataY = pd.read_csv(Utils.getAbsFilePath(train_y_file), index_col=0)
		self.total_data['labels'] = dataY['labels']
		self.lebels = set(dataY['labels'])
		self.X_pred = pd.read_csv(Utils.getAbsFilePath(test_x_file), index_col=0)
		self.totalUniqueWords = set()
		self.wordCounter = defaultdict(int)

	def upDownScale(self, ratio = 10) -> None:
		#Upscale data with lowest frequency or upscale data with highest frequency
		val_count = self.total_data['labels'].value_counts()
		#Is needed if data ratio is like 50:1 or 100:1 we need any data upscale, downscale or augmentation
		#for our data, we are doing upscale - augmentation as highest count is 137 and lowest one is 2, where every categorised data should be at least 137/15 ~ 10
		label_index = defaultdict(list)
		for index, row in self.total_data.iterrows():
			try:
				label_index[self.total_data.at[index, 'labels']].append(index)
			except Exception as err:
				print(f'Error occurred during updating row of train_dna: {err}')
		markerCountNumber = math.ceil(val_count.max()/ratio)
		print("Before Upscale/Augmentation-" + str(self.total_data.shape))
		for label, count in self.total_data['labels'].value_counts().iteritems():
			self.upScale(label_index[label], markerCountNumber-count)
		print("After Upscale/Augmentation- " + str(self.total_data.shape))
		return

	def upScale(self, id_list, no_of_new_data) -> None:
		if no_of_new_data<0: return
		for id_to_augment in random.choices(id_list, k = no_of_new_data):
			#Augmentation is not implemented here, just duplicating data is done in here
			self.total_data = self.total_data.append({
						'labels' : self.total_data.at[id_to_augment, 'labels'],
						'dna' : self.total_data.at[id_to_augment, 'dna']
					},  
					ignore_index = True
				)
		return None

	def clean(self, ifOutlireCharRemoveNeeded = False) -> None:
		self.total_data['dna'] = self.total_data['dna'].apply(lambda x: self.cleanString(x, ifOutlireCharRemoveNeeded))
		self.X_pred['dna'] = self.X_pred['dna'].apply(lambda x: self.cleanString(x, ifOutlireCharRemoveNeeded))
		return

	def preprocess(self, word_len = 4) -> None:
		# Y don't need to be preprocessed because it is already set to numeric values
		self.total_data['dna'] = self.total_data['dna'].apply(lambda x: self.splitWords(x, word_len))
		self.X_pred['dna'] = self.X_pred['dna'].apply(lambda x: self.splitWords(x, word_len))
		print("Max word in a scentence:", self.maxWordLen)
		return

	def save(self, file_name = "input_data.csv", x_test_file_name = "x_test.csv") -> None:
		self.total_data.to_csv(Utils.getAbsFilePath(file_name))
		self.X_pred.to_csv(Utils.getAbsFilePath(x_test_file_name))

	def generateSentenceEmbedding(self) -> None:
		#sbert
		model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
		#For Training Data
		for index, row in self.total_data.iterrows():
			try:
				embedding = model.encode(self.total_data.at[index, 'dna'])
				for embadeIndex, val in enumerate(embedding):
					self.total_data.at[index, "sbert_" + str(embadeIndex).zfill(3)] = val
			except Exception as err:
				print(f'Error occurred during updating row of train_dna: {err}')
		#For Test Data
		for index, row in self.X_pred.iterrows():
			try:
				embedding = model.encode(self.X_pred.at[index, 'dna'])
				for embadeIndex, val in enumerate(embedding):
					self.X_pred.at[index, "sbert_" + str(embadeIndex).zfill(3)] = val
			except Exception as err:
				print(f'Error occurred during updating row of X_pred_dna: {err}')
		self.X_pred.drop(["dna"], axis=1, inplace= True, errors='ignore')
		return

	def generateWord2VecEmbedding(self, vector_size=200, window=20, epochs=100, min_count=2) -> None:
		# Help is in here- https://github.com/Amrit27k/NLPword2vec/blob/master/word2vec-on-rick-morty-dataset.ipynb
		# And help is in here- https://www.linkedin.com/learning/advanced-nlp-with-python-for-machine-learning/how-to-prep-word-vectors-for-modeling?u=87254282
		preprocessed_data = self.total_data['dna'].apply(lambda x: gensim.utils.simple_preprocess(x))
		w2v_model = gensim.models.Word2Vec(preprocessed_data,
								   vector_size=vector_size,
								   window=window,
								   min_count=min_count,
								   epochs=epochs,
								   workers=15,
								   sg=1)

		# Generate a list of words the word2vec model learned word vectors for
		w2v_model.wv.index_to_key
		# Generate aggregated sentence vectors based on the word vectors for each word in the sentence
		w2v_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in w2v_model.wv.index_to_key])
					 for ls in self.total_data['dna']])
		# Compute sentence vectors by averaging the word vectors for the words contained in the sentence
		w2v_vect_avg = []

		#Padding the sequence
		for vect in w2v_vect:
			if len(vect)!=0:
				w2v_vect_avg.append(vect.mean(axis=0))
			else:
				w2v_vect_avg.append(np.zeros(100))

		#Store embedding
		for index, row in self.total_data.iterrows():
			try:
				for embadeIndex, val in enumerate(w2v_vect_avg[index]):
					self.total_data.at[index, "w2vec_" + str(embadeIndex).zfill(4)] = val
			except Exception as err:
				print(f'Error occurred during updating row of w2vec: {err}')

	def generateDoc2VecEmbedding(self, vector_size=200, window=20, epochs=100, min_count=2) -> None:
		#https://www.linkedin.com/learning/advanced-nlp-with-python-for-machine-learning/how-to-implement-doc2vec?u=87254282
		tagged_docs_tr = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(self.total_data['dna'])]

		d2v_model = gensim.models.Doc2Vec(tagged_docs_tr,
								   vector_size=vector_size,
								   window=window,
								   min_count=min_count,
								   epochs=epochs,
								   workers=15)
		#For Train Data
		for index, row in self.total_data.iterrows():
			try:
				embedding = d2v_model.infer_vector(self.total_data.at[index, 'dna'].split())
				for embadeIndex, val in enumerate(embedding):
					self.total_data.at[index, "d2vec_" + str(embadeIndex).zfill(4)] = val
			except Exception as err:
				print(f'Error occurred during updating row of d2vec: {err}')
		#For Test Data
		for index, row in self.X_pred.iterrows():
			try:
				embedding = d2v_model.infer_vector(self.X_pred.at[index, 'dna'].split())
				for embadeIndex, val in enumerate(embedding):
					self.X_pred.at[index, "d2vec_" + str(embadeIndex).zfill(4)] = val
			except Exception as err:
				print(f'Error occurred during updating row of d2vec: {err}')
		self.X_pred.drop(["dna"], axis=1, inplace= True, errors='ignore')
		return

	def generate4MersEncoding(self, vector_size=200, window=20, epochs=100, min_count=2) -> None:
		mlb = MultiLabelBinarizer()
		#self.totalUniqueWords	#=> all columns for words
		for word in self.totalUniqueWords:
			self.total_data['onehot_' + word] = 0
			self.X_pred['onehot_' + word] = 0

		#For Train Data
		df = pd.DataFrame(mlb.fit_transform([x.split(' ') for x in self.total_data["dna"]]),columns=mlb.classes_)
		#df.to_csv(Utils.getAbsFilePath("one_hot.csv"))
		for index, row in self.total_data.iterrows():
			try:
				for word in df.keys():
					self.total_data.at[index, "4mers_" + word] = df.at[index, word]
			except Exception as err:
				print(f'Error occurred during updating row of onehot_train: {err}')

		#For Test Data
		df = pd.DataFrame(mlb.fit_transform([x.split(' ') for x in self.X_pred["dna"]]),columns=mlb.classes_)
		for index, row in self.X_pred.iterrows():
			try:
				for word in df.keys():
					self.X_pred.at[index, "4mers_" + word] = df.at[index, word]
			except Exception as err:
				print(f'Error occurred during updating row of onehot_pred: {err}')
		self.X_pred.drop(["dna"], axis=1, inplace= True, errors='ignore')
		return

	def generateOneHotEncoding(self, vector_size=200, window=20, epochs=100, min_count=2) -> None:
		#Should always done with 1 char and every values
		#So, perWordLength = 1 should be set in .env file for this
		for wordIndex in range(self.maxWordLen):	#For fixing padding
			for word in self.totalUniqueWords:
				self.total_data['onehot_' + str(wordIndex) + word] = 0
				self.X_pred['onehot_' + str(wordIndex) + word] = 0
		#For training Data
		for index, row in self.total_data.iterrows():
			words = self.total_data.at[index, 'dna'].split()
			for i, w in enumerate(words):
				try:
					coumnName = 'onehot_' + str(i) + w
					self.total_data.at[index, coumnName] = 1
				except Exception as err:
					print(f'Error occurred during updating encoding_row_train of onehot_train: {err}')
		#For test Data
		for index, row in self.X_pred.iterrows():
			words = self.X_pred.at[index, 'dna'].split()
			for i, w in enumerate(words):
				try:
					coumnName = 'onehot_' + str(i) + w
					self.X_pred.at[index, coumnName] = 1
				except Exception as err:
					print(f'Error occurred during updating encoding_row_pred of onehot_train: {err}')
		self.X_pred.drop(["dna"], axis=1, inplace= True, errors='ignore')
		return

	def getTrainTestSplit(self, file_name = "input_data.csv", embedding = "sbert"):
		self.total_data = pd.read_csv(Utils.getAbsFilePath(file_name), index_col=0)
		train, test = train_test_split(self.total_data, test_size=0.2)
		train = self.total_data

		y_tr =  train[['labels']]
		y_test = test[['labels']]
		if embedding=="sbert":	#paraphrase-MiniLM-L6-v2_embedding
			X_tr = train[[s for s in train.columns if "sbert_" in s]]
			X_test = test[[s for s in test.columns if "sbert_" in s]]
		elif embedding=="w2vec":
			X_tr = train[[s for s in train.columns if "w2vec_" in s]]
			X_test = test[[s for s in test.columns if "w2vec_" in s]]
		elif embedding == "d2vec":
			X_tr = train[[s for s in train.columns if "d2vec_" in s]]
			X_test = test[[s for s in test.columns if "d2vec_" in s]]
		elif embedding == "4mers":
			X_tr = train[[s for s in train.columns if "4mers_" in s]]
			X_test = test[[s for s in test.columns if "4mers_" in s]]
		elif embedding == "onehot":
			X_tr = train[[s for s in train.columns if "onehot_" in s]]
			X_test = test[[s for s in test.columns if "onehot_" in s]]
		else:
			X_tr = train[['dna']]
			X_test = test[['dna']]
		return (X_tr,y_tr), (X_test,y_test)

	def getXTest(self, file_name = "x_test.csv", embedding = "sbert"):
		self.X_pred = pd.read_csv(Utils.getAbsFilePath(file_name), index_col=0)
		if embedding=="sbert":	#paraphrase-MiniLM-L6-v2_embedding
			X_pred = self.X_pred[[s for s in self.X_pred.columns if "sbert_" in s]]
		elif embedding=="w2vec":
			X_pred = self.X_pred[[s for s in self.X_pred.columns if "w2vec_" in s]]
		elif embedding == "d2vec":
			X_pred = self.X_pred[[s for s in self.X_pred.columns if "d2vec_" in s]]
		elif embedding == "4mers":
			X_pred = self.X_pred[[s for s in self.X_pred.columns if "4mers_" in s]]
		elif embedding == "onehot":
			X_pred = self.X_pred[[s for s in self.X_pred.columns if "onehot_" in s]]
		else:
			X_pred = self.X_pred[['dna']]
		return X_pred
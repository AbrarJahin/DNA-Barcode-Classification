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

class DataCleaning(object):
	#Lambda Functions - Start
	def cleanString(self, text: str)-> str:
		#remove punctuation
		text = "".join([char for char in text if char not in string.punctuation])
		# Convert all to lower
			#not needed in here
		# Remove Stop Words
			#not needed in here
		return text

	def splitWords(self, dna_seq: str, word_len: int)-> str:
		strArray = [dna_seq[index : index + word_len] for index in range(0, len(dna_seq), word_len)]
		text = " ".join(strArray)
		return text
	#Lambda Functions - End

	def __init__(self, train_x_file = "train_features.csv", train_y_file = "train_labels.csv"):
		dataX = pd.read_csv(self.getAbsFilePath(train_x_file), index_col=0)
		dataY = pd.read_csv(self.getAbsFilePath(train_y_file), index_col=0)
		self.total_data = dataX
		self.total_data['labels'] = dataY['labels']
		self.lebels = set(dataY['labels'])

	def getAbsFilePath(self, file_name) -> str:
		script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
		return os.path.join(script_dir, "data/" + file_name)

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

	def clean(self, word_len = 1) -> None:
		self.total_data['dna'] = self.total_data['dna'].apply(lambda x: self.cleanString(x))
		return

	def preprocess(self, word_len = 4) -> None:
		# Y don't need to be preprocessed because it is already set to numeric values
		self.total_data['dna'] = self.total_data['dna'].apply(lambda x: self.splitWords(x, word_len))
		return

	def save(self, file_name = "input_data.csv") -> None:
		self.total_data.to_csv(self.getAbsFilePath("data/" + file_name))

	def generateSentenceEmbedding(self) -> None:
		#sbert
		model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

		for index, row in self.total_data.iterrows():
			try:
				embedding = model.encode(self.total_data.at[index, 'dna'])
				for embadeIndex, val in enumerate(embedding):
					self.total_data.at[index, "sbert_" + str(embadeIndex).zfill(3)] = val
			except Exception as err:
				print(f'Error occurred during updating row of train_dna: {err}')

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
		for index, row in self.total_data.iterrows():
			try:
				embedding = d2v_model.infer_vector(self.total_data.at[index, 'dna'].split())
				for embadeIndex, val in enumerate(embedding):
					self.total_data.at[index, "d2vec_" + str(embadeIndex).zfill(4)] = val
			except Exception as err:
				print(f'Error occurred during updating row of d2vec: {err}')

	def getTrainTestSplit(self, file_name = "input_data.csv", embedding = "sbert"):
		self.total_data = pd.read_csv(self.getAbsFilePath(file_name), index_col=0)
		train, test = train_test_split(self.total_data, test_size=0.2)

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
		else:
			X_tr = train[['dna']]
			X_test = test[['dna']]
		return (X_tr,y_tr), (X_test,y_test)
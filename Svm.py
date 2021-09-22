import keras.backend as K
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten, Reshape, MaxPool1D, BatchNormalization, Dropout, Input
from keras.models import Sequential
from Utils import Utils
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow as tf
import pandas as pd
import datetime
import math
import numpy as np
import keras
#from resnet import Residual
import math
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import sklearn.metrics as metrics
from sklearn import svm

class Svm(object):
	#Lambda Functions - Start
	def recall_m(self, y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision_m(self, y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	#Lambda Functions - End

	def __init__(self, X_tr, y_tr, X_test, y_test, model_filename = 'Svm.sav', epochs = 10, batch_size = 512):
		self.epochs = epochs
		self.batch_size = batch_size

		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test

		self.model_filename = "../model/" + model_filename

		#self.y_tr.shape[1] = 1214

		#Define The Model
		self.model = None

	def trainAndSaveModel(self):
		self.model = svm.SVC(decision_function_shape="ovo").fit(self.X_tr, self.y_tr.values.ravel())
		self.model.predict(self.X_test)
		print("Score -", self.model.score(self.X_test, self.y_test))
		#Save the model
		try:
			pickle.dump(self.model, open(Utils.getAbsFilePath(self.model_filename), 'wb'))	#Store Model to File
		except Exception as e:
			print("SVM-Model Save Failed")
			print(e)
		return

	def restoreModel(self):
		try:
			self.model = pickle.load(open(Utils.getAbsFilePath(self.model_filename), 'rb'))
			result = self.model.score(self.X_test, self.y_test)
			print(result)
			print("Svm-Model Loaded Successfully")
		except Exception as e:
			print("Svm-Model Loaded Failed")
			print(e)

	def savePrediction(self, X_pred, embedding = "sbert", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"_submission.csv"):
		y_pred = self.model.predict(X_pred)
		df = pd.DataFrame({'id':list(X_pred.index),'labels': list(y_pred)})
		df = df.set_index(['id'])
		df.to_csv(Utils.getAbsFilePath(output_file_name))
		return df
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import datetime
import math
from sklearn.metrics import precision_score, recall_score
from Utils import Utils
import pickle
import numpy as np

class RandomForest(object):
	def __init__(self, X_tr, y_tr, X_test, y_test, number_of_trees=500, model_filename = 'RandomForest.sav'):
		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test
		self.model_filename = "../model/" + model_filename
		self.model = None

	def trainAndSaveModel(self):
		rf = RandomForestClassifier()
		self.model = rf.fit(
					self.X_tr,
					self.y_tr.values.ravel()
				)
		pickle.dump(self.model, open(Utils.getAbsFilePath(self.model_filename), 'wb'))	#Store Model to File
		y_pred = self.model.predict(self.X_test)
		precision = precision_score(self.y_test, y_pred, pos_label='positive', average='micro')
		recall = recall_score(self.y_test, y_pred, pos_label='positive', average='micro')
		try:
			accuracy = round((y_pred==self.y_test['label']).sum()/len(y_pred), 3)
			print('Precision: {} / Recall: {} / Accuracy: {}'.format(
					round(precision, 3), round(recall, 3), accuracy))
		except Exception as e:
			print(precision, recall, e)

	def restoreModel(self):
		self.model = pickle.load(open(Utils.getAbsFilePath(self.model_filename), 'rb'))
		result = self.model.score(self.X_test, self.y_test)
		print(result)
		print("RF-Model Loaded Successfully")

	def savePrediction(self, X_pred, embedding = "sbert", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"_submission.csv"):
		y_pred = self.model.predict(X_pred)
		df = pd.DataFrame({'id':list(X_pred.index),'labels': list(y_pred)})
		df = df.set_index(['id'])
		df.to_csv(Utils.getAbsFilePath(output_file_name))
		return df
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import datetime
import math
from sklearn.metrics import precision_score, recall_score

class RandomForest(object):
	def getAbsFilePath(self, file_name) -> str:
		script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
		return os.path.join(script_dir, "data/"+file_name)

	def __init__(self, X_tr, y_tr, X_test, y_test, number_of_trees=500):
		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test
		rf = RandomForestClassifier()
		self.model = rf.fit(
					self.X_tr,
					self.y_tr.values.ravel()
				) #n_estimators=number_of_trees,n_jobs=15,verbose=2,
		y_pred = self.model.predict(X_test)
		precision = precision_score(y_test, y_pred, pos_label='positive', average='micro')
		recall = recall_score(y_test, y_pred, pos_label='positive', average='micro')
		print('Precision: {} / Recall: {} / Accuracy: {}'.format(
			round(precision, 3), round(recall, 3), round((y_pred==y_test['label']).sum()/len(y_pred), 3)))

	def savePrediction(self, test_file_name = "test_features.csv", output_file_name = str(math.ceil(datetime.datetime.now().timestamp()))+"submission.csv"):
		X_pred = pd.read_csv(self.getAbsFilePath(test_file_name), index_col=0)
		y_pred = self.model.predict(X_pred)
		y_pred.to_csv(self.getAbsFilePath("data/" + output_file_name))
		return y_pred
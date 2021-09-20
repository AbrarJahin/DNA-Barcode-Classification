import os

class Utils(object):
	def __init__(self, X_tr, y_tr, X_test, y_test, number_of_trees=500):
		return

	def getAbsFilePath(file_name) -> str:
		script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
		return os.path.join(script_dir, "data/"+file_name)

class Rnn(object):
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

	def __init__(self, X_tr, y_tr, X_test, y_test, number_of_trees=500, model_filename = 'Rnn.sav'):
		self.X_tr, self.y_tr, self.X_test, self.y_test = X_tr, y_tr, X_test, y_test
		self.model_filename = "../model/" + model_filename
		self.model = None

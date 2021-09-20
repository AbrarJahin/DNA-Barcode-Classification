import tensorflow as tf
from DataCleaning import DataCleaning
from RandomForest import RandomForest
import configparser

# Defining main function 
def main():
    config = configparser.ConfigParser()
    config.read('.env')
    #Embedding Config
    isEmbiddingDone = config.get('Default','isEmbiddingDone') == 'True'
    embedding = config.get('Default','embedding') #"sbert", "w2vec" "d2vec"
    perWordLength = int(config.get('Default','perWordLength'))
    outputColumnCount = int(config.get('Default','outputColumnCount')) #Embedding row number
    wordsWindowSize = int(config.get('Default','wordsWindowSize'))
    epochCount = int(config.get('Default','epochCount'))
    minIgnoreCount = int(config.get('Default','minIgnoreCount'))

    #Start Embedding
    dataCleaning = DataCleaning()
    if not isEmbiddingDone:
        dataCleaning.upDownScale()
        dataCleaning.clean()
        dataCleaning.preprocess(perWordLength)
        if embedding=="sbert":
            dataCleaning.generateSentenceEmbedding()
        elif embedding=="w2vec":
            dataCleaning.generateWord2VecEmbedding(vector_size=outputColumnCount, window=wordsWindowSize, epochs=epochCount, min_count=minIgnoreCount)
        elif embedding=="d2vec":
            dataCleaning.generateDoc2VecEmbedding(vector_size=outputColumnCount, window=wordsWindowSize, epochs=epochCount, min_count=minIgnoreCount)
        dataCleaning.save()
  
    (X_tr,y_tr), (X_test,y_test) = dataCleaning.getTrainTestSplit(embedding = embedding)
    #Random Forest
    totalNoOfLebels = len(dataCleaning.lebels)  #Not Needed
    model = RandomForest(X_tr, y_tr, X_test, y_test, totalNoOfLebels)
    model.trainAndSaveModel()
    model.restoreModel()
    model.savePrediction(dataCleaning.X_pred)

# Using the special variable  
# __name__ 
if __name__=="__main__":
    print(tf.__version__)
    main()
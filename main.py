import tensorflow as tf
from DataCleaning import DataCleaning
from RandomForest import RandomForest

# Defining main function 
def main():
    #Embedding Config
    isEmbiddingDone = True
    embedding = "sbert" #"w2vec" "d2vec"
    perWordLength = 4
    outputColumnCount=200 #Embedding row number
    wordsWindowSize=20
    epochCount=100
    minIgnoreCount=2

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
    totalNoOfLebels = len(dataCleaning.lebels)
    model = RandomForest(X_tr, y_tr, X_test, y_test, totalNoOfLebels)
    model.savePrediction("test_features.csv")

# Using the special variable  
# __name__ 
if __name__=="__main__":
    print(tf.__version__)
    main()
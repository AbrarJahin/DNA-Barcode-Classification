# DNA Sequence Classification in Different Ways of Deep Learning

This programme is created as a *text classification* / *DNA sequence classification* problem solving using **TensorFlow** and **Keras**.

## Packages used in this project

List of packages with versions are mentioned in [here](venv_packages.txt). To restore the packages, you need to run this command-

    pip install -r venv_packages.txt

## Run Project (with different configurations)

This project can be run in 2 ways-

    1. By running **main.py** file (with this command- `python3 main.py`)

    2. With Visual Studio (by opening ***DNA Sequence.sln*** file, tasted in [Visual Studio 2019]() )
    2. With Visual Studio (by opening ***DNA Sequence.sln*** file, tasted in [Visual Studio 2019](https://visualstudio.microsoft.com/downloads/ "Visual Studio Download Link") )

All the configurations are coming from a ***.env*** file which is not present in git. To make the file, command is-

    cp .env.example .env

Or just make a copy of **.env.example** file and rename that to **.env**.

in the **.env** file, there are different configurations of the code which can be set form there without changing the code. The ***.env*** file is like this-

```code
[Default]
isEmbiddingDone = False
embedding = onehot
perWordLength = 1
outputColumnCount = 81
wordsWindowSize = 4
epochCount = 10
minIgnoreCount = 2
ifUpscaleNeeded = True
ifCleaningNeeded = True
ifOutlireCharRemoveNeeded = True

isTrainingDone = False
trainingModel = CNN
batchSize = 512
```

So, here are different variables can be set for the code from here. All the possible configurations of the valuse are given in here.

| No. | Variable                  | Possible Values                                                   | Functionality                                                                                                                                                                                                                                                                                   |
|-----|---------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | isEmbiddingDone           | True' or 'False'                                                  | If this value is set to False, then embedding procedure will be activated and will be done depending on *embedding* variable                                                                                                                                                                    |
| 2   | embedding                 | "onehot", "keras", "d2vec", "w2vec" or "sbert"                    | Selection of data encoding and embeddings (after preprocessing)-                                                                                                                                                                                                                                |
|     |                           |                                                                   | 1. *onehot* - For selecting [onehot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) for data (implemented in [here](./DataCleaning.py#227))                                                                                                          |
|     |                           |                                                                   | 2. *keras* - For selecting [keras embedding](https://heartbeat.comet.ml/using-a-keras-embedding-layer-to-handle-text-data-2c88dc019600) where value of k is set from *perWordLength* variable (implemented in [here](./DataCleaning.py#199))                                                    |
|     |                           |                                                                   | 3. *d2vec* - For using [doc to vector embedding](https://medium.com/wisio/a-gentle-introduction-to-doc2vec-db3e8c0cce5e) for data where word length is set from  *perWordLength* variable and window size is set from *wordsWindowSize* variable (implemented in [here](./DataCleaning.py#170)) |
|     |                           |                                                                   | 4. *w2vec* - For using [word to vector] embedding where word length is set from  *perWordLength* variable (implemented in [here](./DataCleaning.py#135))                                                                                                                                        |
|     |                           |                                                                   | 5.*sbert* - Used Pretrained bert which is pretrained on "paraphrase-MiniLM-L6-v2" (implemented in [here](./DataCleaning.py#113))                                                                                                                                                                |
| 3   | perWordLength             | Can be any positive int value                                     | Provides max length of a word during preprocess the dna sequence text                                                                                                                                                                                                                           |
| 4   | outputColumnCount         | Can be any positive int value                                     | No of columns after embeding in "d2vec" or "w2vec". In "sbert", column count is fixed 384                                                                                                                                                                                                       |
| 5   | wordsWindowSize           | Can be any positive int value                                     | Used in *d2vec* for defining window size which needs to be considered during creating embedding                                                                                                                                                                                                 |
| 6   | epochCount                | Can be any positive int value                                     | Total no of epochs for the program                                                                                                                                                                                                                                                              |
| 7   | minIgnoreCount            | Can be any positive int value                                     | Minimum count of the word to be ignored in w2vec                                                                                                                                                                                                                                                |
| 8   | ifUpscaleNeeded           | True' or 'False'                                                  | If we need to upscale the data during preprocessing                                                                                                                                                                                                                                             |
| 9   | ifCleaningNeeded          | True' or 'False'                                                  | If we need to clean the data during preprocessing                                                                                                                                                                                                                                               |
| 10  | ifOutlireCharRemoveNeeded | True' or 'False'                                                  | If we need to remove outlire chars in the character sequence. After outlire removal, only "A", "C", "T", "G"                                                                                                                                                                                    |
| 11  | isTrainingDone            | True' or 'False'                                                  | Is already model trained                                                                                                                                                                                                                                                                        |
| 12  | trainingModel             | "RndomForest", "CNN", "CNN2D", "SVM", "LSTM", "BiLSTM" or "FFNet" | Selection of DL and ML model-                                                                                                                                                                                                                                                                   |
|     |                           |                                                                   | 1. "RndomForest" - For random forest model, implementation can be found in [here](./RandomForest.py)                                                                                                                                                                                            |
|     |                           |                                                                   | 2. "CNN" - For 1D convoloution neural network model, implementation can be found in [here](./Cnn.py)                                                                                                                                                                                            |
|     |                           |                                                                   | 3. "CNN2D" - For 2D convoloution neural network model, implementation can be found in [here](./Cnn2D.py)                                                                                                                                                                                        |
|     |                           |                                                                   | 4. "SVM" - For Support vector machines (SVMs) model, implementation can be found in [here](./Svm.py)                                                                                                                                                                                            |
|     |                           |                                                                   | 5. "LSTM" - For Long short-term memory model, implementation can be found in [here](./Lstm.py)                                                                                                                                                                                                  |
|     |                           |                                                                   | 6. "BiLSTM" - For Bidirectional Long Short-term Memory model, implementation can be found in [here](./BiLstm.py)                                                                                                                                                                                |
|     |                           |                                                                   | 7. "FFNet" - For simple Feed Forward Neural Network, implementation can be found in [here](./FFNet.py)                                                                                                                                                                                          |
| 13  | batchSize                 | Can be any positive int value                                     | No of element per batch for training the model                                                                                                                                                                                                                                                  |
| 14  | augmantationRatio                 | Can be any positive int value                                     | Ratio of Max occured label and min occured elements. So, if any element is occuring less than that, the elements with min occurances are augmented till the ratio are maintained                                                                                                                                                                                         |

All the embedding can work with all the ML model, so different combinations can be possible. All the ML models are self adjustable to the feature size of the embedings (so, if matrix size changes, you don't need to change any value).

If configuration reading failed from `.env` file, then default values are set from [here](./main.py#32).

## Training Status View during Traing

For DL models, tensorboard log is setup from where we can see the training status and learning rates and different graphs.

Command for tensorboard-

    #%tensorboard --logdir logs/fit
    tensorboard --logdir logs
or

    tensorboard --logdir logs/fit

Then go to the browser on the provided url on the command prompt (with port).

    scp ajahin@jamuna.cs.iupui.edu:/home/ajahin/contest/data/x_test.csv ./x_test.csv
    scp ajahin@jamuna.cs.iupui.edu:/home/ajahin/contest/data/input_data.csv ./input_data.csv

Also command prompt output is a good place current training status.

## Data

All the data can be found in [this](./data) folder.

## Model

All the saved models can be found after training in [this](./model) folder.

## Embedding

All the embeddings are saved in [this](./data) folder.

## Some observation on given DNA sequences

### Total Unique Chars-

    ['K', 'R', 'Y', 'M', 'S', '-', 'W', 'N', 'G', 'T', 'C', 'A']

### Occurances of unique words (in train and test)-

    'A': 4100241
    'C': 2195186
    'T': 5107712
    'G': 2010013

    'N': 5541
    'K': 4
    'M': 10
    'R': 18
    'S': 5
    'W': 24
    'Y': 22

So, we can keep only "A", "C", "T", and "G" as others are outlires (which is activated after setting `ifOutlireCharRemoveNeeded` to true).

### Sequence length without punctuation removal

1058

### Max sequence length with punctuation removal

744

### Augmentation (with ratio set to 10)

Before Upscale/Augmentation-(12906, 2)
After Upscale/Augmentation- (18543, 2)

## Followed Procedure

1. Read data in pandas
2. Preprocess
    1. If upscale needed (which is needed for our case), upscale/ augment the data
    2. Removed punctuation and outlire charecters
    3. Then process the seequence to chars of defined length each by inserting required saces in required places
    4. Create embeddings/encodings for feeding ML/DL models
    5. Pad the data embedding for making all the data having same no of features
    6. Split the data with training and test for training and validating the model
3. Feed the encoding/embedding to the ML model
4. Validate scores
5. Save the model and predictions

Saved predictions can be found in [this folder](./data/) with name like `*_submission.csv`.

## Command needed to run the program in Linux Server

    cd DNA-Barcode-Classification
    python3 main.py

If you need to run in background detached from current user with writing the output to a file, then the command should be like this-

    nohup python3 ~/DNA-Barcode-Classification/main.py>~/output.txt &

And then see the current output like this-

    tail -f ~/output.txt
___________

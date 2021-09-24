# DNA Sequence Classification in DIfferent Ways

Run Command-

cp .env.example  .env
python3 main.py

All configurations are saved in .env file. So, before every run, you need to tweek the .env file for different embeddings.

## See Log by command-

    %tensorboard --logdir logs/fit
    tensorboard --logdir logs

Then go to the browser on the provided url (with port).

    scp ajahin@jamuna.cs.iupui.edu:/home/ajahin/contest/data/x_test.csv ./x_test.csv
    scp ajahin@jamuna.cs.iupui.edu:/home/ajahin/contest/data/input_data.csv ./input_data.csv

##Total Unique Words-

['K', 'R', 'Y', 'M', 'S', '-', 'W', 'N', 'G', 'T', 'C', 'A']

ACTG 

##Word length without punctuation removal

1058

##Word length with punctuation removal

744*11 = 

## Augmentation-

Before Upscale/Augmentation-(12906, 2)
After Upscale/Augmentation- (18543, 2)

## Get a model layer input shape-

model.layers[0].input_shape
model.layers[0].output_shape

### check if input shape are OK
i == 0....n-1
model.layers[i].output_shape == model.layers[i+1].input_shape

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

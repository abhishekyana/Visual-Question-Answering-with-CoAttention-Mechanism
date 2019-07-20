#To run the code

Simple BaseLine Model:

1) python SimpleNet_main.py
The Code will be executed and Tensorboard scalars are written.

CoAttention Model:

1) Download the GloVe.6B.300d.txt file using 
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d ./supportfiles/
```
2) Run the `CoAttention_preprocess.py` to generate the pickle objects required to run the model experiment runner.
3) python CoAttention_main.py

Later, We can see the graphs in the ./RUNS folder

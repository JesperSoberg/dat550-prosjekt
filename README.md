# Bag of Words Document Classification using Feedforward Neural Network and Recurrent Neural Network

This document will provide a simple guide to train and test neural networks on your own using the code provided here. A list of the available parameters will be provided with a brief explenation at the bottom of this document.

# Improtant after cloning!

All of the training and test data is available within this repo under the Archive folder. However, they are compressed and will therefore have to be decompressed for anything to run. The decompressed files should be stored in the same location and under the same name (minus the .gz) as the compressed files. The training data for example should be under Archive/arxiv_train.csv

## Training and testing a neural network to predict label of abstracts

0. (OPTIONAL) If you want to pick the parameters that a neural network should use, you may go in the "experiments.json" file and add a new configuration similar to the ones already present. However, do be careful about choosing certain values, as they have been observed to impact the runtime greatly. These will be marked with a "*" in the parameter list. The configurations within the file on GitHub contain the configuration of the final results presented in our report. Thus, simply running main.py with these configurations should produce similar results. 

1. Once you've created your own configurations (or skipped to this part), you can run the main.py file either through the code editor of your choice or running it through the terminal.

2. If you want the results to be logged in your wandb.ai profile, you can change the project and entity variables in the "wandb.init()" call in main.py where it is marked as <INSERT PROFILE NAME HERE> and <INSERT ENTITY NAME HERE>. 

3. When main.py has successfully executed, it will print the final measurement metrics in the terminal. Additionally, if you have done the previous step you may also look at them in wandb.ai. These aforementioned measurements are: Accuracy, Precision, Recall, F1-score and Loss.

## Parameter list for the neural networks:

* **name**: The name of the neural network.
#### num_epochs: The amount of epoch to run the training and testing for.
#### bow: Which bag of words method you want the network to use, the options being "tf_idf" and "countVector".
#### network: The type of network you want to use. Options: ["ffnn", "rnn"].
#### nrows: How many rows of the dataset you want to include in the run.
#### hidden_size: Exclusive to RNN, determines how large the hidden layer is.
#### num_layers: Exclusive to RNN, determines how many RNNs stacked on top of each other.
#### num_hidden_layers: Exclusive to FFNN, determines the amount of hidden layers for FFNN.
#### learning_rate: Which learning rate the optimizer will be using.
#### batch_size: How large the batches will be each iteration during training and testing.
#### load_path: If there is already a trained model, you can specify its path for this parameter and you will skip the training process. 
#### save_path: Should you want to save a run/model, you may specify where you want to save it here. Use either a ".pt" or ".pth" file extension when writing a save path.


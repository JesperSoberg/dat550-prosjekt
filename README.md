# Bag of Words Document Classification using Feedforward Neural Network and Recurrent Neural Network

## Training and testing a neural network to predict abstract label

### In order to do a run on you own you can go in the experiments.json file and add a new experiment like the ones that are already present. However do be careful about nrow values over 1'000 as as it can impact the running time greatly on less powerful machines, especially for running FFNNs.

## The parameters you can adjust are the following:
##
## name: The name of the neural network.
## num_epochs: The amount of epoch to run the training and testing for.
## bow: Which bag of words method you want the network to use, the options being "tf_idf" and "countVector".
## network: The type of network you want to use. Options: ["ffnn", "rnn"].
## nrows: How many rows of the dataset you want to include in the run.
## hidden_size: Exclusive to RNN, determines how large the hidden layer is.
## num_layers: Exclusive to RNN, determines how many RNNs stacked on top of each other.
## num_hidden_layers: Exclusive to FFNN, determines the amount of hidden layers for FFNN.
## learning_rate: Which learning rate the optimizer will be using.
## batch_size: How large the batches will be each iteration during training and testing.
## load_path: If there is already a trained model, you can specify its path for this parameter and you will skip the training process. 
## save_path: Should you want to save a run/model, you may specify where you want to save it here. Use either a ".pt" or ".pth" file extension when writing a save path.
##
## Once you've created your experiments or skipped that part. You can run the main.py file either through your code editor of choice or running it in the terminal.
##
## If you want to add it to your wand.ai profile you can change the project and entity variables in the "wandb.init()" call so that you can look at the graphs that is created from your run on your own.
##
## When the program has fully executed it will print the final measurement metrics, those being: Accuracy, Precision, Recall, Macro F1-score and Loss.
##
##
##

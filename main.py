import wandb
import json
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from dataSet import CustomDataDataSet
from rnn import RNN
from ffnn import FFNN
from Preprocessing import getDataFrameFromData
from BoW import TF_IDF, getCountVector
from networkFunctions import train, test

def run_experiment(parameter_dict):
    wandb.init(
        project="RNNs and You",
        config=parameter_dict,
        name=parameter_dict["name"]
    )

    train_df, train_labels = getDataFrameFromData("Archive/arxiv_train.csv", nrows=parameter_dict["nrows"])
    test_df, test_labels = getDataFrameFromData("Archive/arxiv_test.csv", nrows=parameter_dict["nrows"])

    if parameter_dict["bow"] == "tf_idf":
        train_tensors, vocabulary = TF_IDF(train_df)
        test_tensors, _ = TF_IDF(test_df, vocabulary=vocabulary)
    elif parameter_dict["bow"] == "countVector":
        train_tensors, vocabulary = getCountVector(train_df)
        test_tensors, _ = getCountVector(test_df, vocabulary=vocabulary)
    else:
        return
    
    train_dataset = CustomDataDataSet(train_tensors, train_labels)
    test_dataset = CustomDataDataSet(test_tensors, test_labels)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=parameter_dict["batch_size"],
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=parameter_dict["batch_size"],
                                 shuffle=True)
    
    if parameter_dict["network"] == "rnn":
        print(train_tensors.shape[1])
        model = RNN(train_tensors.shape[1],
                    parameter_dict["hidden_size"],
                    parameter_dict["num_layers"])
    elif parameter_dict["network"] == "ffnn":
        model = FFNN(train_tensors.shape[1])
    else:
        return

    if parameter_dict["load_path"] is not None:
        try:
            model.load_state_dict(torch.load(parameter_dict["load_path"]))
        except FileNotFoundError:
            return
        
    loss_function = nn.CrossEntropyLoss()

    if parameter_dict["load_path"] is None:
        optimiser = torch.optim.Adam(model.parameters(),
                                     lr=parameter_dict["learning_rate"])
  
        for _ in range(parameter_dict["num_epochs"]):
            train(train_dataloader, model, optimiser, loss_function)
            test(test_dataloader, model, loss_function)

        torch.save(model.state_dict(), parameter_dict["save_path"])
    
    else:
        for _ in range(parameter_dict["num_epochs"]):
            test(test_dataloader, model, loss_function)

    wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(888)
    np.random.seed(888)

    with open("experiments.json") as file:
        experiments = json.load(file)

    for experiment in experiments:
        run_experiment(experiment)
import wandb
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataSet import CustomDataDataSet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Preprocessing import getVectorLabels



def finalPrediction(predictions):
    result = []
    for prediction in predictions:
        final_prediction = [0]*10
        index = prediction.argmax()
        final_prediction[index] = 1
        result.append(final_prediction)
    return torch.tensor(result)


def evaluate(predictions, labels):
	accuracy = accuracy_score(y_true=labels, y_pred=predictions)
	precision = precision_score(y_true=labels, y_pred=predictions, average="macro", zero_division=0)
	recall = recall_score(y_true=labels, y_pred=predictions, average="macro", zero_division=0)
	f1 = f1_score(y_true=labels, y_pred=predictions, average="macro", zero_division=0)

	return accuracy, precision, recall, f1


def train(dataloader, model, optimiser, loss_function):
	model.train()

	num_batches = len(dataloader)
	training_loss = 0

	for X, y in dataloader:

		predictedLabels = model(X)
		loss = loss_function(predictedLabels, y)
		training_loss += loss.item()

		loss.backward()
		optimiser.step()
		optimiser.zero_grad()

	training_loss /= num_batches
	wandb.log({"Training loss": training_loss})


def test(dataloader, model, loss_function):
	model.eval()
	
	num_batches = len(dataloader)
	test_loss = 0

	all_predictions = torch.tensor([])
	all_labels = torch.tensor([])
	with torch.no_grad():
		for X, y in dataloader:
			predictions = model(X)
			test_loss += loss_function(predictions, y).item()
			predictions = finalPrediction(predictions)
			all_predictions = torch.cat((all_predictions, predictions), 0)
			all_labels = torch.cat((all_labels, y), 0)
			

	accuracy, precision, recall, f1 = evaluate(all_predictions, all_labels)
	test_loss /= num_batches
	
	wandb.log({"Accuracy": accuracy, "Precision": precision, "Recall": recall, "Macro-f1-score": f1, "Test loss": test_loss})

	
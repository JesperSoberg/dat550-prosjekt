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

	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	batch_size = num_batches/size

	for batch, (X, y) in enumerate(dataloader):

		predictedLabels = model(X)
		loss = loss_function(predictedLabels, y)

		loss.backward()
		optimiser.step()
		optimiser.zero_grad()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * batch_size + len(X)
			print(f"loss: {loss}  [{current}/{size}]")


def test(dataloader, model, loss_function):
	model.eval()
	
	num_batches = len(dataloader)
	test_loss = 0

	with torch.no_grad():
		for X, y in dataloader:
			predictions = model(X)
			test_loss += loss_function(predictions, y).item()
			predictions = finalPrediction(predictions)
			accuracy, precision, recall, f1 = evaluate(predictions, y)

	test_loss /= num_batches
	
	print(f"Test Error: \n Accuracy: {accuracy}, Precision: {precision}, recall: {recall}, f1: {f1}, Avg loss: {test_loss} \n")

	
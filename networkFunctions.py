import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataSet import CustomDataDataSet



def finalPrediction(predictions):
    result = []
    for prediction in predictions:
        final_prediction = [0]*10
        index = prediction.argmax()
        final_prediction[index] = 1
        result.append(final_prediction)
    return torch.tensor(result)


def train(trainModel, optimizer="adam", data=None, labels=None):
	#print("Training...")
	numEpochs = 5
	learnRate = 0.01
	
	dataSet = CustomDataDataSet(data, labels)

	optimizer = torch.optim.Adam(trainModel.parameters(), lr=learnRate)
	lossFunction = nn.CrossEntropyLoss()#BCELoss()

	batchSize = 50
	TrainDataLoader = DataLoader(dataSet, batch_size=batchSize, shuffle=True)
	for epoch in range(numEpochs):
		print(f"Epoch {epoch+1}\n-------------------------------")
		size = len(TrainDataLoader.dataset)
		for batch, (X, y) in enumerate(TrainDataLoader):
			optimizer.zero_grad()
			predictedLabels = trainModel(X)

			loss = lossFunction(predictedLabels, y)

			loss.backward()
			optimizer.step()
			if batch % 100 == 0:
				loss, current = loss.item(), batch * batchSize + len(X)
				print(f"loss: {loss}  [{current}/{size}]")
		print(f"loss: {loss}  [{current}/{size}]")

	return trainModel
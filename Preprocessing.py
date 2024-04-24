import string
import torch
import pandas as pd

def cleanAbstract(abstract):
	abstract = abstract.lower()
	translator = str.maketrans("\n", " ", string.punctuation)
	return abstract.translate(translator)

def getVectorLabels(dataframe):
	labelLookup = {
		"eess": 	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		"quant-ph": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		"physics": 	[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		"stat": 	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		"math": 	[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		"astro-ph": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		"cond-mat": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		"hep-th": 	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		"cs": 		[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		"hep-ph": 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
	}
	result = []
	for label in dataframe["label"]:
		result.append(labelLookup[label])

	return torch.tensor(result, dtype=torch.float)


def getDataFrameFromData(data, nrows=None):
	dataFrame = pd.read_csv(data, nrows=nrows)
	labels = getVectorLabels(dataFrame)

	dataFrame = dataFrame.map(lambda cell: cleanAbstract(str(cell)))

	return dataFrame, labels
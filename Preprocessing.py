import string
import torch

def cleanAbstract(abstract):
	abstract = abstract.lower()
	translator = str.maketrans("\n", " ", string.punctuation)
	return abstract.translate(translator)

def sortFrameByID(dataFrame):
	return dataFrame.sort_values(by=['id'])

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
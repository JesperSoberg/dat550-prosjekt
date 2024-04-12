from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix

import numpy as np
import torch


def getCountVector(df, vocabulary=None):
    vectoriser = CountVectorizer(vocabulary=vocabulary)
    vector = vectoriser.fit_transform(df["abstract"])
    tensor = countVector2Tensor(vector)
    return tensor, vectoriser.vocabulary_

def countVector2Tensor(vector):
	coo = coo_matrix(vector)

	values = coo.data
	indices = np.vstack((coo.row, coo.col))

	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = coo.shape

	tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
	return tensor

def TF_IDF(dataFrame, column):
	vectorizer = TfidfVectorizer(stop_words=['an', 'it', 'in', 'and', 'all', 'and', 'was', 'the', 'of', 'more', 'than',
			   'are', 'for', 'to', 'which', 'is', 'its', 'that', 'two', 'when',
			   'our'])
	
	X = vectorizer.fit_transform(dataFrame[column])

	Y = vectorizer.get_feature_names_out()
	return X, Y


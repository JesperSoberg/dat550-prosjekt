from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix

import numpy as np
import torch


def getCountVector(df, vocabulary=None):
    vectoriser = CountVectorizer(vocabulary=vocabulary)
    vector = vectoriser.fit_transform(df["abstract"])
    tensor = csrMatrix2Tensor(vector)
    return tensor, vectoriser.vocabulary_

def csrMatrix2Tensor(matrix):
	coo = coo_matrix(matrix)

	values = coo.data
	indices = np.vstack((coo.row, coo.col))

	i = torch.LongTensor(indices)
	v = torch.FloatTensor(values)
	shape = coo.shape

	tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense()
	return tensor

def TF_IDF(dataFrame, vocabulary=None):
	W2Rm = ['an', 'it', 'in', 'and', 'all', 'and', 'was', 'the', 'of', 'more', 'than',
			   'are', 'for', 'to', 'which', 'is', 'its', 'that', 'two', 'when',
			   'our', 'this', 'be']
	vectorizer = TfidfVectorizer(stop_words=W2Rm, vocabulary=vocabulary)
	
	X = vectorizer.fit_transform(dataFrame['abstract'])
	torchTensor = csrMatrix2Tensor(X)
	
	return torchTensor, vectorizer.vocabulary_


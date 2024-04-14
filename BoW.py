from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import coo_matrix

import numpy as np
import torch
import pandas as pd
from Preprocessing import printDocumentWords


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

	tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
	return tensor

def TF_IDF(dataFrame, labels, vocabulary=None):
	W2Rm = ['an', 'it', 'in', 'and', 'all', 'and', 'was', 'the', 'of', 'more', 'than',
			   'are', 'for', 'to', 'which', 'is', 'its', 'that', 'two', 'when',
			   'our', 'this', 'be']
	vectorizer = TfidfVectorizer(stop_words=W2Rm, vocabulary=vocabulary)
	
	X = vectorizer.fit_transform(dataFrame['abstract'])

	#Y = vectorizer.get_feature_names_out()

	
	# WordDocDF = WordDocumentDataFrame
	#WordDocDF = pd.DataFrame.sparse.from_spmatrix(X).T.set_index(Y)

	# Index = Ordene (som dukker opp i minst ett dokument)
	# Column = Hvilket dokument det er, feks. column=50 er det førtiniende dokumentet i dataframen
 
 	# threshH er hvor stor treshhold for printDocumentWords
	# DocToView er hvilket dokument en vil se på for printDocumentWords
	threshH = 0.07
	DocToView = 0

	#printDocumentWords(WordDocDF, docIdx=DocToView, threshhold=threshH, labels=labels)

	torchTensor = csrMatrix2Tensor(X).to_dense()
	
	return torchTensor, vectorizer.vocabulary_


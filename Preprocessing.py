import string

def cleanAbstract(abstract):
	abstract = abstract.lower()
	translator = str.maketrans("\n", " ", string.punctuation)
	return abstract.translate(translator)

def sortFrameByID(dataFrame):
	return dataFrame.sort_values(by=['id'])



#Threshhold er hvor høy TF_IDF scoren må være for å bli printet
def printDocumentWords(WordDocDF, docIdx, threshhold, labels):
	print(labels[docIdx])
	for i in range(WordDocDF.shape[docIdx]):
		if WordDocDF[docIdx].iloc[i] > threshhold:
			label = WordDocDF.iloc[[i]].index.values[docIdx]
			print(f"Word: {label}, TF_IDF: {WordDocDF[0].iloc[i]}")


def dropEnglishWords(dataFrame, X, Y, wordsToRemove):
	for i in range(X.shape[1]):
	#print(Y[i])
		if Y[i] in wordsToRemove:
			dataFrame = dataFrame.drop(index=Y[i])
	return dataFrame
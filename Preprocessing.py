import string

def cleanAbstract(abstract):
	abstract = abstract.lower()
	translator = str.maketrans("\n", " ", string.punctuation)
	return abstract.translate(translator)

def sortFrameByID(dataFrame):
	return dataFrame.sort_values(by=['id'])
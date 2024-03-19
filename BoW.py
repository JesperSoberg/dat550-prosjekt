from sklearn.feature_extraction.text import TfidfVectorizer

def TF_IDF(dataFrame):
	vectorizer = TfidfVectorizer()
	
	X = vectorizer.fit_transform(dataFrame['abstract'])

	Y = vectorizer.get_feature_names_out()
	return X, Y
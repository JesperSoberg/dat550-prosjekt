from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def getCountVector(df):
    vectoriser = CountVectorizer()
    vector = vectoriser.fit_transform(df["abstract"])
    return vector

def TF_IDF(dataFrame, column):
	vectorizer = TfidfVectorizer(stop_words="english")
	
	X = vectorizer.fit_transform(dataFrame[column])

	Y = vectorizer.get_feature_names_out()
	return X, Y


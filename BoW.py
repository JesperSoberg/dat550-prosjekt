from sklearn.feature_extraction.text import CountVectorizer


def getCountVector(df):
    vectoriser = CountVectorizer()
    vector = vectoriser.fit_transform(df["abstract"])
    return vector

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data
corpus = [
    'This is a good example of text.',
    'We are learning NLP using Python.',
    'Text data needs preprocessing before using ML models.'
]

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer()

# Fit and transform the corpus
result = tfidf.fit_transform(corpus)

# Get indexing (vocabulary)
print('\nWord indexes:')
print(tfidf.vocabulary_)

# Display tf-idf values
print('\nTF-IDF value (sparse matrix):')
print(result)

# In matrix form
print('\nTF-IDF values in array/matrix form:')
print(result.toarray())


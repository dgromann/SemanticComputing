# TU Dresden, SS 2018, Dagmar Gromann
# Starting point for Naive Bayes algorithm

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

#To speed up processing, we retrieve a subset of the 20 topics in the dataset
categories =  ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print("The obtained categories are the ones provided: ", twenty_train.target_names)
print("Can be addressed using data: ", len(twenty_train.data))
print("The file names are available: ", len(twenty_train.filenames))

print("First line of the first loaded file: ", "\n".join(twenty_train.data[0].split("\n")[:3]))

#Target attributes are stored as list of integers
print(twenty_train.target[:10])

#To print the name of the category:
for topic in twenty_train.target[:10]:
    print(twenty_train.target_names[topic])

#Let's use a function of sklearn to tokenize the text and filter out stop words and return a dictionary of features where
#documents are transformed to feature vectors
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(twenty_train.data)
print("Number of documents and length of vocabulary ", X_train_counts.shape)

#In this dictionary we can address specific words where the index value is linked to its frequency in the whole training corpus
print("Frequency of algorithm ", count_vec.vocabulary_.get('algorithm'))

#One problem: longer documents will have higher counts when we only look at occurrences than shorter ones
#Solution: we divide the number of occurrences of each word in a document by the total number of words in the document = term frequency
#then we downsclae the weights for words that occur in many documents using tf-idf (term frequency inverse document frequency)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

#TODO: choose a naive bayes classifier from http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#TODO: do the same pipeline to process the test data and then calculate the accuraccy of the clasifier using np.mean or something similar


twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

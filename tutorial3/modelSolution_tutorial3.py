# TU Dresden, SS 2018, Dagmar Gromann
# Starting point for Naive Bayes algorithm

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import numpy as np
import matplotlib.pyplot as plt

#Gaussian Naive Bayes expects a normal distribution of all features so it is suboptimal for text data and classification
#Since word occurrences are sparse it is better to use the Multinomial or Bernoulli NB

#To speed up processing, we retrieve a subset of the 20 topics in the dataset
categories =  ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print("The retrieved categories are the ones we restricted the dataset to, which are: ", twenty_train.target_names, "\n")
print("Data can be addressed using \"twenty_train.data\": ")
print(twenty_train.data, "\n")
print("The file names are available: ")
print(twenty_train.filenames, "\n")

print("First line of the first loaded file: ", "\n".join(twenty_train.data[0].split("\n")[:3]), "\n")

#Target attributes are stored as list of integers
print("One target vector takes the form of:")
print(twenty_train.target[:10])

#To print the name of the category:
print("The above target vector represents the following categories in numeric format:")
for topic in twenty_train.target[:10]:
    print(twenty_train.target_names[topic])

#Let's use a function of sklearn to tokenize the text and filter out stop words and return a dictionary of features where
#documents are transformed to feature vectors
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(twenty_train.data)
print("\n", "Number of documents and length of vocabulary: ", X_train_counts.shape)

#In this dictionary we can address specific words where the index value is linked to its frequency in the whole training corpus
print("\n", "Frequency of the word \"algorithm\" in the created vocabulary ", count_vec.vocabulary_.get('algorithm'))

#One problem: longer documents will have higher counts when we only look at occurrences than shorter ones
#Solution: we divide the number of occurrences of each word in a document by the total number of words in the document = term frequency
#then we downsclae the weights for words that occur in many documents using tf-idf (term frequency inverse document frequency)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("\n",  "Shape of the training data after: ", X_train_tfidf.shape)

#TODO: choose a naive bayes classifier from http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#TODO: do the same pipeline to process the test data and then calculate the accuraccy of the clasifier using np.mean or something similar

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
X_test_counts = count_vec.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

#Multinomial NB captures word frequency information and estimates the conditional probability of a term belonging
#to a specific class as the relative frequency of the term in documents belonging to that particular class
# It uses the relative frequency of a term in the documents of the training set of a specific class to calculate the
#probability of it belonging to that specific class
#Conditional probability = the fraction of tokens/positions in documents of class c that contain term t
multinomial = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
mn_predicted = multinomial.predict(X_test_tfidf)
print("\n",  "Accuracy of Multinomial NB", np.mean(mn_predicted == twenty_test.target))

#Bernoulli NB creates a Boolean indicator for each term that is set to 1 if the term is in the document and 0 if it is absent, while the
#Multinomial NB only takes present words into consideration and not absent ones. Bernoulli NB ignores the number of occurrences, while the
#Multinomial NB also takes multiple occurrences into account. This means that it is more likely to fail for the classification of longer documents.
#For example, it might assign a whole long document to Tokyo because of one occurrence of the word.
#Conditional probability = fraction of documents of class c that contain term t
bernoulli = BernoulliNB().fit(X_train_tfidf, twenty_train.target)
bn_predicted = bernoulli.predict(X_test_tfidf)
print("\n",  "Auccuracy of BernoulliNB", np.mean(bn_predicted == twenty_test.target))


feature_names = count_vec.get_feature_names()
feature_names = np.asarray(feature_names)

print("\n", "Top 10 keywords per class:")
for i, category in enumerate(categories):
    top10 = np.argsort(multinomial.coef_[i])[-10:]
    print("%s: %s" % (category, " ".join(feature_names[top10])))

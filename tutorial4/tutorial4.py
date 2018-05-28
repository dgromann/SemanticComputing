# TU Dresden, SS 2018, Dagmar Gromann
# Supervised machine learning algorithms


import numpy as np
import nltk

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

#To speed up processing, we retrieve a subset of the 20 topics in the dataset
categories =  ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#TODO: Prepare the data creating bag of words and their frequency count (CountVectorizer)
#TODO: Reduce the weight of very common words by using TF-IDF


#TODO: Run the algorithm of your choice - SVM or Decision Tree

#TODO: Calculate the accuarcy for the algorithm of your choice
print("Accuracy of LinearSVC", np.mean(your_classifier == twenty_test.target))

#TODO: Compare the accuracy of your algorithm to a Naive Bayes baseline
print("Accuracy of Multinomial NB", np.mean(multinomialNB_classifier == twenty_test.target))


#TODO: Fine-Tune your Parameters using GriDSearchCV - for this you have to check the specific parameters for each classifier and choose the onse that you believe require adapation
print("Best score achieved with fine-tuning SGD:", GriDSearchCV_classifier.best_score_)
print("Best parameter settings for SGD: ", GriDSearchCV_classifier.best_params_)


#TODO: Test if the accuracy changes with Lemmatization, Stemming, or stop word removal
def lemmatizing_tokenizer(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

#Stemming
def stemming_tokenizer(text):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w) for w in word_tokenize(text)]

count_vec_stem = CountVectorizer(tokenizer=oneOfTheAboveMethods)
#TODO: fill in the preprocessing and the training steps for the algorithm of your choice
print("Accuraccy of stemmed SVC:", np.mean(your_stemmed/lemmatized_classifier == twenty_test.target))
print()

#TODO: Remove stop words and test accuracy again

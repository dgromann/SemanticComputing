# TU Dresden, SS 2018, Dagmar Gromann
# Supervised learning algorithms

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
tfidf_vec = TfidfVectorizer()
X_train_tfidf = tfidf_vec.fit_transform(twenty_train.data)
X_test_tfidf = tfidf_vec.transform(twenty_test.data)


#TODO: Run the algorithm of your choice - SVM or Decision Tree
#Different SVMs for classification
svm = LinearSVC().fit(X_train_tfidf, twenty_train.target)
svm_predicted = svm.predict(X_test_tfidf)

nu_svc = NuSVC().fit(X_train_tfidf, twenty_train.target)
nu_svc_predicted = nu_svc.predict(X_test_tfidf)

svc = SVC().fit(X_train_tfidf, twenty_train.target)
svc_predicted = svc.predict(X_test_tfidf)

#Example of how to run all steps in one go in a "pipeline"
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, max_iter=5, random_state=42))])
sgd_predicted = text_clf_svm.fit(twenty_train.data, twenty_train.target).predict(twenty_test.data)

#Decision Tree
dt = DecisionTreeClassifier(min_samples_split=40).fit(X_train_tfidf,twenty_train.target)
dt_predicted = dt.predict(X_test_tfidf)

random_forest = RandomForestClassifier().fit(X_train_tfidf, twenty_train.target)
rf_predicted = random_forest.predict(X_test_tfidf)

#TODO: Calculate the accuarcy for the algorithm of your choice
print("Accuracy of LinearSVC", np.mean(svm_predicted == twenty_test.target))
print("Accuracy of NuSVC", np.mean(nu_svc_predicted == twenty_test.target))
print("Accuracy of SVC", np.mean(svc_predicted == twenty_test.target))
print("Accuracy of SGD", np.mean(sgd_predicted == twenty_test.target))

print("Accuracy of Decision Tree", np.mean(dt_predicted == twenty_test.target))
print("Accuracy of Random Forest", np.mean(rf_predicted == twenty_test.target))

#TODO: Compare the accuracy of your algorithm to a Naive Bayes baseline
nb_multinomial = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
mn_predicted = nb_multinomial.predict(X_test_tfidf)
print("Accuracy of Multinomial NB", np.mean(mn_predicted == twenty_test.target))
print()

#TODO: Fine-Tune your Parameters using GriDSearchCV - for this you have to check the specific parameters for each classifier and choose the onse that you believe require adapation
parameters_sgc = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_sgc)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)
print("Best score achieved with fine-tuning SGD:", gs_clf_svm.best_score_)
print("Best parameter settings for SGD: ", gs_clf_svm.best_params_)
print()

parameters_svc = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
gs_clf_svc = GridSearchCV(svc, parameters_svc)
gs_clf_svc = gs_clf_svc.fit(X_train_tfidf, twenty_train.target)
print("Best score achieved with fine-tuning SVC:", gs_clf_svc.best_score_)
print("Best parameter settings for SVC: ", gs_clf_svc.best_params_)
print()

param_dt = {'min_samples_split': [2, 20, 40]}
gs_clf_dt = GridSearchCV(dt, param_dt)
gs_clf_dt = gs_clf_dt.fit(X_train_tfidf, twenty_train.target)
print("Best score achieved with fine-tuning Decision Trees:", gs_clf_dt.best_score_)
print("Best parameter settings for Decision Trees: ", gs_clf_dt.best_params_)
print()

#TODO: Test if the accuracy changes with Lemmatization, Stemming, or stop word removal
def lemmatizing_tokenizer(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

#Stemming
def stemming_tokenizer(text):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w) for w in word_tokenize(text)]

count_vec_stem = CountVectorizer(tokenizer=stemming_tokenizer)
tfidf_stem = TfidfTransformer()
twenty_train_stem = count_vec_stem.fit_transform(twenty_train.data)
twenty_train_tfidf = tfidf_stem.fit_transform(twenty_train_stem)

twenty_test_stem = count_vec_stem.transform(twenty_test.data)
twenty_test_tfidf = tfidf_stem.transform(twenty_test_stem)

svm = LinearSVC().fit(twenty_train_tfidf, twenty_train.target)
svm_predicted_stem = svm.predict(twenty_test_tfidf)
print("Accuraccy of stemmed LinearSVC:", np.mean(svm_predicted_stem == twenty_test.target))

count_vec_lem = CountVectorizer(tokenizer=lemmatizing_tokenizer)
tfidf_lem = TfidfTransformer()
twenty_train_lem = count_vec_lem.fit_transform(twenty_train.data)
twenty_train_tflem = tfidf_lem.fit_transform(twenty_train_stem)

twenty_test_lem = count_vec_lem.transform(twenty_test.data)
twenty_test_tflem = tfidf_lem.fit_transform(twenty_test_stem)

svm = LinearSVC().fit(twenty_train_tflem, twenty_train.target)
svm_predicted_stem = svm.predict(twenty_test_tflem)
print("Accuraccy of lemmatized LinearSVC:", np.mean(svm_predicted_stem == twenty_test.target))


text_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=stemming_tokenizer)),
                         ('tfidf', TfidfTransformer()),
                         ('svm', SVC(C=10,kernel="linear"))])
svc_predicted_stem = text_clf_svm.fit(twenty_train.data, twenty_train.target).predict(twenty_test.data)
print("Accuraccy of stemmed SVC:", np.mean(svc_predicted_stem == twenty_test.target))


sgd_clf_svm = Pipeline([('vect', CountVectorizer(tokenizer=stemming_tokenizer)),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, max_iter=5, random_state=42))])
sgd_predicted_stem = sgd_clf_svm.fit(twenty_train.data, twenty_train.target).predict(twenty_test.data)
print("Accuraccy of stemmed SGD:", np.mean(sgd_predicted_stem == twenty_test.target))

rfc = Pipeline([('vect', CountVectorizer(tokenizer=stemming_tokenizer)),
                         ('tfidf', TfidfTransformer()),
                         ('rfc', RandomForestClassifier())])
rfc_predicted_stem = rfc.fit(twenty_train.data, twenty_train.target).predict(twenty_test.data)
print("Accuraccy of stemmed Random Forests:", np.mean(rfc_predicted_stem == twenty_test.target))
print()


#Stop word removal
svc_stop = Pipeline([('vect', CountVectorizer(stop_words="english")),
                         ('tfidf', TfidfTransformer()),
                         ('svm', SVC(C=10,kernel="linear"))])
svc_predicted_stop = svc_stop.fit(twenty_train.data, twenty_train.target).predict(twenty_test.data)
print("Accuraccy of stop word removed SGD:", np.mean(svc_predicted_stop == twenty_test.target))


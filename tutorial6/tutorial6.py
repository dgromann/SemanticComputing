# TU Dresden, SS 2018, Dagmar Gromann
# Word Embeddings

import gensim
import re
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from matplotlib import pyplot


# Load the subset of the word2vec pre-trained embedding library trained on the Google News corpus
model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_embeddings.bin', binary=True)

#EXERCISE 1:
#This is how you obtain a specific vector for a word
#The whole vocabulary of the embedding library can be obtained with model.wv.vocab.keys()
print(model["good"])

#TODO: get the length of the vector and its datatype


#This is how you can get the 10 (10 top number of words: topn=10) most similar words for a specific word (here: good)
most_similar = model.most_similar("good", topn=10)
print("Top 10 most similar words to \"good\": ", most_similar)

#This is how you can try the analogy task of a is to b as c is to? Here it says king - man + woman or positives(king, woman), negative(man)
analogy = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print("Man is to woman as king is to: ", analogy)

#TODO: rewrite the analogy function using addition and subtraction of vevtors and then query the word for the vector using "model.similar_by_vector(vector)"
#The vector in brackets is the input to the function and the vector you created



# #EXERCISE 2:
# # Load the "20_newsgroups" corpus from the sklearn library if datasets
# twenty_news = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers','footers','quotes'))
#
# #TODO: Process the loaded dataset to obtain a list of sentences where each sentence represents a list of words
# #The following regular expression can for instance be used to split each text in the twenty_news corpus into sentences:
# re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
#
# #Outcomment the following line to train the model with your sentences
# news_model = Word2Vec(sentences, size=100, window=5, min_count=5)

# #TODO: how long is your vocabulary in the model? Is it really words?
#
# #Outcomment the following line to save the model to your disk to re-use later and below
# news_model.wv.save_word2vec_format("news_model.bin", binary=True)


# #EXERCISE 3:
# #TODO: Comment out the code for the above two exercises and load the model you trained in Exercise 2
# news_model = gensim.models.KeyedVectors.load_word2vec_format('20newsgroup.bin', binary=True)
#
#  #TODO: Load the file analogy.txt and evaluate your embeddings and the model embeddings on this dataset by predicting the last word in each line
#  # You input the first three words in the correct order into the analogy task and then check whether the top 3 predictions contain the correct answer
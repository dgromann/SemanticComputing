import os
import gensim
import re
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from matplotlib import pyplot

def visualize_embeddings(model):
    pca = PCA(n_components=2)
    result = pca.fit_transform(model[model.wv.vocab])
    pyplot.scatter(result[:,0], result[:,1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words[:20]):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


# Load the subset of the word2vec pre-trained embedding library trained on the Google News corpus
model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_embeddings.bin', binary=True)

#EXERCISE 1:
#This is how you obtain a specific vector for a word
#The whole vocabulary of the embedding library can be obtained with model.wv.vocab.keys()
print(model["good"])

#TODO: get the length of the vector and its datatype
print("Number of dimensions ", len(model["good"]))
print("Data type ", type(model["good"]))

#This is how you can get the 10 (10 top number of words: topn=10) most similar words for a specific word (here: good)
most_similar = model.most_similar("good", topn=10)
print("Top 10 most similar words to \"good\": ", most_similar)

#This is how you can try the analogy task of a is to b as c is to? Here it says king - man + woman or positives(king, woman), negative(man)
analogy = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print("Man is to woman as king is to: ", analogy)
print("Types of relations: antonymy, partial synonymy, superlative (better)")

#TODO: rewrite the analogy function using addition and subtraction of vevtors and then query the word for the vector using "model.similar_by_vector(vector)"
#The vector in brackets is the input to the function and the vector you created
analogy_2 = model.similar_by_vector((model["woman"]+model["king"]) - model["man"], topn=3)
print(analogy_2)


# #EXERCISE 2:
# # Load the "20_newsgroups" corpus from the sklearn library if datasets
# twenty_news = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers','footers','quotes'))
#
# #TODO: Process the loaded dataset to obtain a list of sentences where each sentence represents a list of words
# sentences = []
# for text in twenty_news.data:
#     for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text):
#         sentences.append(word_tokenize(sentence))
#
#
# #Train the model with your sentences
# news_model = Word2Vec(sentences, size=100, window=5, min_count=5)
# # #size = number of dimensions for the embeddings
# # window = maximum distance between a center word and its context words
# # min_count = minimum counts of words to consider when training the model
#
# #TODO: how long is your vocabulary in the model? Is it really words?
# print(news_model.wv.vocab.keys())
# print(len(news_model.wv.vocab.keys()))
#
# # Save the model to your disk to re-use later and below
# news_model.wv.save_word2vec_format("20newsgroup.bin", binary=True)


#EXERCISE 3:
#TODO: Comment out the code for the above two exercises and load the model you trained in Exercise 2
news_model = gensim.models.KeyedVectors.load_word2vec_format('20newsgroup.bin', binary=True)

#You can use this method to visualize the first 100 words in your embeddings if you want
#visualize_embeddings(news_model)

# #TODO: Load the file analogy.txt and evaluate your embeddings and the model embeddings on this dataset by predicting the last word in each line
# # You input the first three words in the correct order into the analogy task and then check whether the top 3 predictions contain the correct answer

evalFile = open("data/analogy.txt")
evalLines = evalFile.readlines()

corr = 0
corr_fake = 0
for line in evalLines:
    if not ": " in line:
        first = line.split()[0]
        second = line.split()[1]
        third = line.split()[2]
        answer = line.split()[3]
        results = model.most_similar(positive=[second, third], negative=[first], topn=1)
        results_fake = news_model.most_similar(positive=[second, third], negative=[first], topn=1)
        if answer in results[0][0]:
            corr += 1
        if answer in results_fake[0][0]:
            corr_fake += 1

print(corr/len(evalLines))
print(corr_fake/len(evalLines))


evalFile.close()

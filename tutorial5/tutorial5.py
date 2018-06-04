
import re
import string
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


directory = "enron/"


def parse_email(directory):
    headers = {}
    body = {}
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if name.endswith('.txt'):
                f = open(os.path.join(dirpath, name))
                lines = f.readlines()

        #TODO: process the above lines to find the features from and to in the headers and the body of the mail into the body
        #Feel free to change the data type if you wish to use another type than dictionaries to store the extracted features
    return headers, body

headers, body = parse_email(filepath)
print("Number of e-mails: ", len(headers), len(body))


#TODO: Convert body to matrix of TF-IDF features

#TODO: change the following variable "X" to the name of your matrix and plot the results
X_dense = X.todense()
coords = PCA(n_components=2).fit_transform(X_dense)

plt.scatter(coords[:, 0], coords[:, 1], c='m')
plt.show()

#TODO: use your TF-IDF feature matrix to get the 20 top keywords of all e-mails (function "name of your TfidfVectorizer".get_feature_names() and then

#TODO: Perform clustering on the data using k-means

#TODO: Plot the resulting clusters using the function plot_clustering_results (see above)
plot_clustering_results(..., ... )

#TODO: get the top TF-IDF features for each cluster

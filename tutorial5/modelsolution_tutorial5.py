# TU Dresden, SS 2018, Dagmar Gromann
# Unsupervised machine learning algorithms

import re
import string
import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward
from random import randint

filepath= "enron/"

# Some Regular Expresssions for processing e-mails
# Finding different types of linebreaks
linebreak = ["\n", "\r", "\r\n"]  # unix, mac, and DOS newlines

# Finding header lines
header_re = re.compile('([^:]+):\s(.*)')


#One possible version of how to parse e-mails
def parse_email(directory):
    from_dict = {}
    to_dict = {}
    emails_body = {}
    for dirpath, dirnames, files in os.walk(directory):
        for name in files:
            if name.endswith('.txt'):
                f = open(os.path.join(dirpath, name))
                lines = f.readlines()

                body = ""
                doneWithHeader = False
                originalMessage = False
                headers = {}
                for line in lines:
                    if doneWithHeader == False:
                        if header_re.match(line) != None or ":" in line:
                            if "From: " in line:
                                from_dict[name[:-4]] = line.split(":")[1].replace("\n", "")
                            if "To: " in line:
                                to_dict[name[:-4]] = line.split(":")[1].replace("\n", "")
                        if line in linebreak:
                            doneWithHeader = True
                    elif "Original Message" in line or "Forwarded by" in line:
                        originalMessage = True
                    elif originalMessage == True:
                        if line in linebreak:
                            originalMessage = False
                    else:
                        line = re.sub(r'[^a-zA-Z]', ' ', line)
                        body += line
                emails_body[name[:-4]] = body
    return emails_body, from_dict, to_dict


def plot_clustering_results(X, clf, n_clusters):
    # Let's plot this with matplotlib to visualize it.
    # First we need to make 2D coordinates from the sparse matrix.
    X_dense = X.todense()
    pca = PCA(n_components=2).fit(X_dense)
    coords = pca.transform(X_dense)

    # Lets plot it again, but this time we add some color to it.
    # This array needs to be at least the length of the n_clusters.
    label_colors = []
    for i in range(n_clusters):
        label_colors.append('#%06X' % randint(0, 0xFFFFFF))
    colors = [label_colors[i] for i in labels]

    plt.scatter(coords[:, 0], coords[:, 1], c=colors)
    # Plot the cluster centers
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d60')
    plt.title("Visualization of k-means clustering results ")
    plt.show()


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df


def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []

    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

#Exercise 1:
print("A possible solution to exercise 1: ")
#TODO: process the above lines to find the features from and to in the headers and the body of the mail into the body
print("There is a python native, built in library for parsing emails, which is ``from email.parser import Parser´´")
print("It can be called using ``Parser().parsestr(lines)´´ or ```email.message_from_string()´´. Howver, it is important to understand those kind of ")
print("data processing steps, which is best done by implementing it ourselves at least once")
print()
body, from_dict, to_dict = parse_email(filepath)
print("Length of from dict: ", len(from_dict), "Length of to dict: ", len(to_dict), "Length of body dict: ", len(body))
print()

#TODO: Convert body to matrix of TF-IDF features
vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(body.values())
print("Vocabulary of TfidfVectorizer without stop words: ", vect.vocabulary_)

#TODO: change the following variable "X" to the name of your matrix and plot the results
X_dense = X.todense()
coords = PCA(n_components=2).fit_transform(X_dense)
plt.title("Visualization data using PCA")
plt.scatter(coords[:, 0], coords[:, 1], c='m')
plt.show()

#TODO: use your TF-IDF feature matrix to get the 20 top keywords of all e-mails (function "name of your TfidfVectorizer".get_feature_names() and then
feature_names = np.array(vect.get_feature_names())
#argsort only supports small to large sorting; to change this order we use -1 in the following line
sort = np.argsort(vect.idf_)
top_n = 20
print("This is a little extra code to get some better understanding for TF-IDF")
print("These are the features with the lowest Inverse Document Frequency (IDF; Document Frequencies are \"penalized\" with IDF for appearing in many emails, i.e., made lower)")
print((feature_names[sort[:top_n]]))
print("These are the features with the highest IDF, so they are likely to appear in very few if not only one email: ")
print(feature_names[sort[-top_n:]])
print()

max_values = X.max(axis=0).toarray().ravel()
sort_by_tfidf = max_values.argsort()
print("Maximum TF-IDF values throughout the dataset: ")
print(feature_names[sort_by_tfidf[-top_n:]])

print("Values with the lowest TF-IDF values:")
print(feature_names[sort_by_tfidf[:top_n]])
print()

#Exercise 2
#TODO: Perform clustering on the data using k-means
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
labels = kmeans.fit_predict(X)

#TODO: Plot the resulting clusters using the function plot_clustering_results (see above)
plot_clustering_results(X, kmeans, n_clusters)

#TODO: get the top TF-IDF features for each cluster
print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
for i in range(n_clusters):
    print("Cluster %d:" % i),
    top_features = ""
    for ind in order_centroids[i, :top_n]:
        top_features += feature_names[ind] + ","
        #print(' %s' % feature_names[ind]),
    print(top_features)





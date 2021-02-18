from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

import csv
import numpy as np
from numpy import ndarray
import pandas as pd
from time import time
from scipy.sparse import csr_matrix

from os import path
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

#read data from a CSV file
sampled_file = "/home/giulia/Downloads/distro/Distro/sampled_file.csv"
df =pd.read_csv(sampled_file)
mat=df[df.columns[0]].to_numpy()
dataset = mat.tolist()
mat=df[df.columns[1]].to_numpy()
labels = mat.tolist()
true_k = len(np.unique(labels))

#vectorization
vectorizer = TfidfVectorizer()
for i in range(len(dataset)):
    dataset[i]=np.str_(dataset[i])
X = csr_matrix(vectorizer.fit_transform(dataset))
Y = X.toarray()
print(X.shape)
print("The original data have", X.shape[1], "dimensions/features/terms")

#compute PCA
#d_std = preprocessing.StandardScaler().fit_transform(Y)
pca=PCA(n_components=true_k)
d_pca=pca.fit_transform(Y)
#d_pca is a numpy array with transformed data
 
v=pca.explained_variance_ratio_
print(v)

#kmeans 
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(d_pca)
print("done in %0.3fs" % (time() - t0))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
    % metrics.adjusted_rand_score(labels, km.labels_))
acc = metrics.accuracy_score(labels, km.labels_)
if (acc<0.5): acc = 1- acc
print("Accuracy: %0.3f" % acc) 
print(km.cluster_centers_.shape)

original_centroids = pca.inverse_transform(km.cluster_centers_)
print(original_centroids.shape) 

for i in range(original_centroids.shape[0]):
    original_centroids[i] = np.array([x for x in original_centroids[i]])
pca_centroids = original_centroids.argsort()[:, ::-1]

text0= []
text1 = []
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in pca_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        if i==0 : text0.append(terms[ind])
        if i == 1: text1.append(terms[ind])
    print()

# Create and generate a word cloud image for cluster 0:
wordcloud = WordCloud(background_color="white").generate(' '.join(text0))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#generate a word cloud image for cluster 1:
wordcloud = WordCloud(background_color="white").generate(' '.join(text1))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
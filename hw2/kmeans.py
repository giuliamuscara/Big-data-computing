from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans
import csv
import numpy as np
import pandas as pd
from time import time

from os import path
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator

lemmatized_file = "/home/giulia/Downloads/distro/Distro/lemmatized_file.csv"
df =pd.read_csv(lemmatized_file)
mat=df[df.columns[0]].to_numpy()
dataset = mat.tolist()
mat=df[df.columns[1]].to_numpy()
labels = mat.tolist()
true_k = len(np.unique(labels))

#vectorization
vectorizer = TfidfVectorizer()
for i in range(len(dataset)):
    dataset[i]=np.str_(dataset[i])
X = vectorizer.fit_transform(dataset)
print(X.shape)
print("The original data have", X.shape[1], "dimensions/features/terms")

print("Clustering with Kmeans:")

#Clustering with kmeans
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

acc = metrics.accuracy_score(labels, km.labels_)
if (acc<0.5): acc = 1- acc
print("Accuracy of kmeans: %0.3f" % acc)

text0 = []
text1 = []
centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        if i==0: text0.append(terms[ind])
        else: text1.append(terms[ind])
    print()

# Create and generate a word cloud image for cluster 0:
wordcloud = WordCloud(background_color="white").generate(' '.join(text0))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#generate wordcloud for cluster 1:
wordcloud = WordCloud(background_color="white").generate(' '.join(text1))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("Now we perform the clustering with MiniBatchKmeans:")

#Clustering with mini batch kmeans
km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', batch_size=100 , max_iter=100)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    
acc = metrics.accuracy_score(labels, km.labels_)
if (acc<0.5): acc = 1- acc
print("Accuracy of MiniBatchKmeans: %0.3f" % acc) 

text0 = []
text1 = []
centroids = km.cluster_centers_.argsort()[:, ::-1] ## Indices of largest centroids' entries in descending order
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in centroids[i, :10]:
        print(' %s' % terms[ind], end='')
        if i == 0 : text0.append(terms[ind])
        else: text1.append(terms[ind])
    print()

# Create and generate a word cloud image for cluster 0:
wordcloud = WordCloud(background_color="white").generate(' '.join(text0))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#generate wordcloud for cluster 1:
wordcloud = WordCloud(background_color="white").generate(' '.join(text1))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
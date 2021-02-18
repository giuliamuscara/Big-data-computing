from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.cluster import KMeans
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

r = true_k
t0 = time()
svd = TruncatedSVD(r)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
Y = lsa.fit_transform(X)
print("done in %fs" % (time() - t0))
var = 0
print(svd.explained_variance_ratio_)
print("The number of documents is still", Y.shape[0])
print("The number of dimension has become", Y.shape[1])

terms = vectorizer.get_feature_names()
print("The most relevant terms after svd are:")
for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    s = ""
    for t in sorted_terms:
        s += t[0] + " "
    print(s)

#kmeans 
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100)
t0 = time()
km.fit(Y)
print("done in %0.3fs" % (time() - t0))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
    % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
    % metrics.silhouette_score(Y, km.labels_, sample_size=1000))
acc = metrics.accuracy_score(labels, km.labels_)
if (acc<0.5): acc = 1 - acc
print("Accuracy: %0.3f" % acc) 
print(km.cluster_centers_.shape)

original_centroids = svd.inverse_transform(km.cluster_centers_)
print(original_centroids.shape) 

for i in range(original_centroids.shape[0]):
    original_centroids[i] = np.array([x for x in original_centroids[i]])
svd_centroids = original_centroids.argsort()[:, ::-1]

print("The most relevant terms after svd and kmeans are:")

text0= []
text1 = []
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in svd_centroids[i, :10]:
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
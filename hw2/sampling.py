from random import random
import csv
import scipy.stats as st
from math import ceil

lemmatized_file = "/home/giulia/Downloads/distro/Distro/lemmatized_file.csv"
sampled_file = "/home/giulia/Downloads/distro/Distro/sampled_file1.csv"

def sample(N, cl, e, p):
    # first we get the z-score
    z = 1.96 #z-score is approximately 1.96 for 95% of confidence
    # then the n_0 value
    n_0 = z**2 * p * (1 - p) / e**2
    # and finally we calculate n
    n = n_0 / (1 + (n_0 / N))
    # we also need to round up to the nearest integer
    n = ceil(n)
    # finally we return our sample size
    return n

csvf = open(lemmatized_file)
reader = csv.reader(csvf)
nlines = len(list(reader))
sample_size = sample(nlines, 0.95, 0.01, 0.5)
print("The sample size to get 0.95 of confidence level is %d" % sample_size)
csvf.close()

# Setting sample rate
rate = ceil((100*sample_size)/nlines) 
buckets = 100 # This is used for sampling

print("Sampling rate:", rate, "(%)")
csvf = open(lemmatized_file)
reader = csv.DictReader(csvf)
# hash users if you want to sample fraction <rate> of users
with open(sampled_file,"w",newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["line","label"])
    for row in reader:
        text = row['line']
        hash_value = hash(text)%buckets
        if hash_value < rate:
            writer.writerow([row['line'], row['label']])

sampled_csvf = open(sampled_file)
reader = csv.reader(sampled_csvf)
lines = len(list(reader))
print("Number of records in the sampled lemmatized csv:", lines)
sampled_csvf.close()
csvf.close()



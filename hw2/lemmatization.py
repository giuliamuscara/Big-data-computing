import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet, stopwords 
from sklearn.preprocessing import Normalizer
import csv
import numpy as np
import pandas as pd
from string import punctuation

original_file = "/home/giulia/Downloads/distro/Distro/csv_file.csv"
lemmatized_file = "/home/giulia/Downloads/distro/Distro/lemmatized_file.csv"
df =pd.read_csv(original_file)
mat=df[df.columns[0]].to_numpy()
dataset = mat.tolist()
mat=df[df.columns[1]].to_numpy()
labels = mat.tolist()

#lemmatizion with WordNet and pos tags
lemmatizer = nltk.WordNetLemmatizer()

# POS_TAGGER_FUNCTION : TYPE 1 
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None

stop_words = set(stopwords.words('english'))
punteggiatura = set(punctuation)
stop_words = stop_words.union(punteggiatura)
lista = ['lot','lol','like','love', 'good', 'great','make', 'dr','fy','ok', 'na', 'ha', 'one' ,'two', 'three', 'four','five', 'six', 'aka', 'ala', 'get', 'well', 'would', 'could', 'vs' ,'nt','etc', 'ca', 'mth', 'mae', 'oh' , 'ah', 'yo', 'rd', 'mr', 'alot', 'ft', 'haha', 'yr', 'bc', 'wa', 'wo', 'oc','didnt', 'dont', 'doesnt', 'gonna']
stop_words =stop_words.union(lista)

with open(lemmatized_file,"w",newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["line","label"])
    for i in range(len(dataset)):
        dataset[i] = dataset[i].replace('-',' ')
        dataset[i] = dataset[i].replace('/',' ')
        dataset[i] = dataset[i].replace('.',' ')
        word_list = nltk.word_tokenize(dataset[i])
        pos_tagged = nltk.pos_tag(word_list)
        wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
        lemmatized_doc = ""
        for word, tag in wordnet_tagged: 
            word = word.lower()
            if word.isalpha() and len(word) > 1 and not word in stop_words:
                if tag is None: 
                    # if there is no available tag, append the token as is 
                    lemmatized_doc =  lemmatized_doc + " " + lemmatizer.lemmatize(word)
                else:         
                    # else use the tag to lemmatize the token 
                    lemmatized_doc = lemmatized_doc + " " + lemmatizer.lemmatize(word, tag)
        if len(lemmatized_doc) > 0:
            writer.writerow([lemmatized_doc, labels[i]])

print("Lemmatization executed on the whole file")

#check the number of lines is correct
csvf = open(lemmatized_file)
reader = csv.reader(csvf)
lines = len(list(reader)) 
print("Number of records in the lemmatized csv file:", lines)
csvf.seek(0)
csvf.close()

    
        




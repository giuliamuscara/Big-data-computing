from random import random
import csv

original_file = "/home/giulia/Downloads/distro/Distro/corpus.txt"
labels_file = "/home/giulia/Downloads/distro/Distro/labels.txt"
csv_file = "/home/giulia/Downloads/distro/Distro/csv_file.csv"
f_orig = open(original_file, "r")
l_origin = open(labels_file, "r")

with open(csv_file,"w",newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["line","label"])
    for line in f_orig:
        writer.writerow([line, l_origin.readline()])

f_orig.close()
f_orig = open(original_file, "r")
# Counting total number of lines
# Just out of curiosity
nlines = sum(1 for _ in f_orig)
print("Number of records in the original files:", nlines)
f_orig.seek(0)

csvf = open(csv_file)
reader = csv.reader(csvf)
lines = len(list(reader))
print("Number of records in the csv file (one line is reserved for the headers):", lines)
csvf.close()
f_orig.close()
l_origin.close()
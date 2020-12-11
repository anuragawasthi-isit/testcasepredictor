# Import all the dependencies
import multiprocessing
import tfsconnect
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from main import testcaselst, acptcrtlst, y, clean_text, lemmatizer
import csv
import _multiprocessing
import math
import string
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from gensim.models import word2vec
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pandas as pd
import html2text
from os import listdir

num_cores = multiprocessing.cpu_count()
#"C:\\tcdataextract\\DSATCDATAV2.csv" dsa data file
tcdatafilename = "C:\\tcdataextract\\tcdata.csv"
df = pd.read_csv(tcdatafilename, encoding='utf-8')
#read acceptance criteria directly from tfs
#print(clean_text(tfsconnect.getacptnccriteria(2875301)).replace("give","").replace("when","").replace("user",""))
df['tcsteps'] = df['tcsteps'].fillna("NA").apply(clean_text)
df['tctitle'] = df['tctitle'].fillna("NA")
df['TCID'] = df['TCID'].fillna("NA")
#df['AcptCrit'] = df['AcptCrit']
# df['TCID'] = df['TCID'].apply(clean_text)
TESTTITLE = df.tctitle
YTCID = df.TCID
ZTCSTEPS = df.tcsteps
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
TESTTTLEDATA = TESTTITLE.tolist()
TCIDdata = YTCID.tolist()
data = ZTCSTEPS.tolist()

coupleddata = dict(zip(TCIDdata, data))


# Python program to convert a list to string

# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = "  "

    # traverse in the string
    for ele in s:
        str1 += " " + ele

    # return string
    return str1

#Tagged the test steps and tags are testcase ids the code tokenize the words and create documents
#tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(data)]
# tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[enumerate(data)]) for  _d in enumerate(data)]
tagged_data = [TaggedDocument(words=word_tokenize(capital.lower()), tags=[str(state).replace(".0","")]) for state, capital in coupleddata.items()]
# print(type(tagged_data))
# f = open("demofile2.txt", "a")
# for item in tagged_data:
#     f.write(str(item.words)+"===="+str(item.tags))
# f.close()
max_epochs = 210
vec_size = 2000
alpha = 0.025
#
model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=4,
                dm=1)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
#basic1 is dsa model
model.save("dellcom.model")
# print("Model Saved")

model = Doc2Vec.load("dellcom.model")

# to find the vector of a document which is not in training data
# test_data = word_tokenize(acptcrtlst[0].lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1')
# print(similar_doc)
print(acptcrtlst[0])
tokens = [clean_text(str(acptcrtlst[0]))]
print(listToString(tokens))
# tokens = tokens.remove("Given")

new_vector = model.infer_vector(listToString(tokens).split())
# sims = model.docvecs.most_similar([new_vector])
# print(sims)
most_similar_docs = []
# for d in model.docvecs.most_similar([new_vector]):
#     print(int(d[0]))
#     most_similar_docs.append(coupleddata[d[0]])
MOST_SIMILAR_TCS =model.docvecs.most_similar([new_vector])

similar_tcs = [tc_tuple[0] for tc_tuple in MOST_SIMILAR_TCS]
for item in similar_tcs:
    print(item+"===="+tfsconnect.getworkitemtitle(item))
# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])

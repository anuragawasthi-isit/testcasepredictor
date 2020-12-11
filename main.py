import csv
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
from nltk.stem.snowball import SnowballStemmer
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

from unicodecsv import writer
lemmatizer = nltk.WordNetLemmatizer()
englishStemmer = SnowballStemmer("english")
directory = "C:\\acptdataextract"
filename = "C:\\acptdataextract\\ACTCMAPPING.csv"

tcdatafilename = "C:\\tcdataextract\\TCDATA.csv"

df = pd.read_csv("C:\\acptdataextract\\ACTCMAPPING.csv", encoding='cp1252')
df['AcptCrit'] = df['AcptCrit'].fillna("NA")
df['TCsteps'] = df['TCsteps'].fillna("NA")
# print(df.head(10))
#print(df['AcptCrit'].apply(lambda x: len(x.split(' '))).sum())


def print_plot(index):
    example = df[df.index == index][['AcptCrit', 'TCID']].values[0]
    if len(example) > 0:
        print(example[0])
        print('TCID:', example[1])


def converthtmlfilestotxt():
    i = 1
    for filename in listdir(directory):
        path = directory + '\\' + filename
        html = open(path, encoding="utf8")
        f = html.read()
        w = open(filename, "w")
        b = html2text.html2text(f)
        w.write(str(b))
        html.close()
        w.close()
        i += 1

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
def lemmatize_text(text):
    a = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return str(a)

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [englishStemmer(w) for w in tokens]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text)  # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    text = ' '.join(word for word in text.split() if len(word) > 2)
    text = ' '.join(word for word in text.split() if word.isalpha())
    text = ' '.join(englishStemmer.stem(word) for word in text.split())
    text = ' '.join(lemmatizer.lemmatize(word, pos="v") for word in text.split() if word.isalpha())

    return text
def avg_sentence_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec




df['AcptCrit'] = df['AcptCrit'].apply(clean_text)
df['TCsteps'] = df['TCsteps'].apply(clean_text)
# print_plot(10)
# print(df['AcptCrit'].apply(lambda x: len(x.split(' '))).sum())
X = df.AcptCrit
y = df.TCsteps
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(y_train)
# X_train =X_train.fillna('NA')
print(X_train.shape)
# y_train=y_test.fillna('NA')
print(y_train.shape)
testcaselst =[]
acptcrtlst =[]
for items in y_train:
    items = list(items.split(" "))
    testcaselst.append(items)

for items in X:
    items = list(items.split(" "))
    acptcrtlst.append(items)
#print(acptcrtlst[0])

#print(testcaselst)
#print(X_train.tolist())
# train model
word2vec_model = word2vec.Word2Vec(testcaselst, min_count=1)
# summarize the loaded model
#print(word2vec_model)
# summarize vocabulary
words = list(word2vec_model.wv.vocab)
#print(words)
# access vector for one word
#print(word2vec_model['dsa'])
# save model
#word2vec_model.save('model.bin')
# fit a 2d PCA model to the vectors
X = word2vec_model[word2vec_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
#pyplot.scatter(result[:, 0], result[:, 1])
words = list(word2vec_model.wv.vocab)
# X_test =X_test.fillna('NA')
# y_test=y_test.fillna('NA')
# nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#                ])
# nb.fit(X_train, y_train)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()





#get average vector for sentence 1

sentence_1_avg_vector = avg_sentence_vector(X_train[0].split(), model=word2vec_model, num_features=100,index2word_set=word2vec_model.wv)

#get average vector for sentence 2

sentence_2_avg_vector = avg_sentence_vector(y_train[0].split(), model=word2vec_model, num_features=100,index2word_set=word2vec_model.wv)
senta=sentence_1_avg_vector.reshape(-1,1)
sentb=sentence_2_avg_vector.reshape(-1,1)
sen1_sen2_similarity =  cosine_similarity(senta,sentb)

# y_pred = nb.predict(X_test)

# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred))


# for iem in aa['AcptCrit']:
#     if str(type(iem))=='float':
#         print("null")
#     else:
#         aaa = html2text.html2text(iem)
#         newlst.append(aaa)

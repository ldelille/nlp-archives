#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:00:06 2021

@author: eva
"""

### Imports ###

from scipy.spatial.distance import cdist  # a row, + a matrix

import pandas as pd
import numpy as np
import pickle
import time
from joblib import load

from gensim.models.fasttext import FastText

import spacy

from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()

nlp = spacy.load("fr_core_news_sm")

from spacy.lang.fr import French

parser = French()

try:
    from nltk.corpus import stopwords
except:
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords

import re

### Preprocessing ###

## Stopwords ##

fr_stop = set(stopwords.words('french'))
# v1 : numbers
my_fr_stop = fr_stop.union({'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix',
                            'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
                            'vingt', 'trente', 'quarante', 'cinquante', 'soixante', 'cent'}, fr_stop)
# v2 : conj + det + verbs
my_fr_stop = fr_stop.union({'ce', 'celui', 'cette', 'cet', 'celui-là', 'celui-ci',
                            'le', 'la', 'les', 'de', 'des', 'du',
                            'mais', 'où', 'et', 'donc', 'or', 'ni', 'car', 'depuis', 'quand', 'que', 'qui', 'quoi',
                            'ainsi', 'alors', 'avant', 'après', 'comme',
                            'être', 'avoir', 'faire',
                            'autre'})

nlp = spacy.load("fr_core_news_sm")


## Tokenizer ##

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif '@' in str(token):
            lda_tokens += str(token).split('@')
        else:
            lda_tokens.append(token.lower_)
    return [t for t in lda_tokens if len(str(t)) > 0]


## Stemmer ##

def prepare_text_stem(text):
    """
    Input:
    ------
    text: string, raw text
    
    Output:
    ------
    tokens: list of string, tokenized, filtered and lemmatized words from the input text
    """
    tokens = tokenize(text)  # split and lower case
    tokens = [re.sub(r'\b\d+\b', '', token).strip(' ') for token in tokens]  # get rid of digits
    tokens = [token for token in tokens if len(token) > 3]  # arbitrary length, +get rid of empty strings
    tokens = [token for token in tokens if token not in my_fr_stop]  # stopwords
    # print("Remaining tokens : ", tokens)
    tokens = [stemmer.stem(token) for token in tokens]  # obtain lemmas
    return tokens


def preprocess(news_df):
    """
    news_df must be a dataframe with text and title columns
    """
    title_tokens = []
    text_tokens = []

    ## Apply on titles ##

    for t in news_df.title:
        tokens = prepare_text_stem(t)
        title_tokens.append(tokens)

    ## Apply on bodies ##

    for t in news_df.text:
        tokens = prepare_text_stem(t)
        text_tokens.append(tokens)

    # save the preprocessed text
    news_df["clean_text"] = [' '.join(text_tokens[i]) for i in range(len(text_tokens))]
    news_df["clean_title"] = [' '.join(title_tokens[i]) for i in range(len(title_tokens))]

    return news_df


### Load data and models ###

start = time.time()

model = FastText.load("fasttext_finetuned_50K_120d.model")  # finetuned fasttext embedding, 120d vectors

sample_df = pd.read_excel("sample_articles.xlsx")
lemonde_df = pd.read_csv("../recommendation_models/archives_ready.csv")  # one level above current folder

with open('../recommendation_models/tfidf_vectorizer_50K', 'rb') as handle:  # tfidf model
    tfidf_vectorizer = pickle.load(handle)

# preload reduced embeddings, standard scaler and PCA objects
utils = dict()
for n in [4, 8, 12]:
    utils[n] = dict()
    utils[n]["embedding"] = np.loadtxt("./E_" + str(n) + "_archives_reduced.txt")
    utils[n]["std_scaler"] = load("std_scaler_50K_" + str(n) + ".joblib")
    utils[n]["pca"] = load("pca_50K_" + str(n) + ".joblib")
print("\n>>> Pre-loading  <<< Elapsed: ", (time.time() - start))

### Preprocessing ###

sample_df = preprocess(sample_df)


### Utils ###

## Apply tfidf + FastText embedding on top n words ###

def embed_top_n(news_df, tfidf_vectorizer, n, model):
    """
    Input
    -----
    news_df : Pandas Dataframe, must contain a "clean_text" column.
    tfidf_vectorizer : already fitted tfidf vectorizer 
    n : number of top tfidf words we want to keep for the embedding
    model : fine-tuned fasttext embedding using gensim  instead of fasttext pre-trained and preloaded model. 
    Previsouly hard-coded here : 300-dimensional embedding. Now 120d. 
    
    Output
    -----
    E : np.array, size # examples * # embedding dimensions. 
    Obtained by averaging the embeddings of the top words selected using tfidf.
    """
    # Apply tfidf to the dataset
    X = tfidf_vectorizer.transform(news_df.clean_text)  # X is a matrix
    # print(X.shape)
    # get features of tfidf
    feature_array = np.array(tfidf_vectorizer.get_feature_names())

    # extract top n words
    N = X.shape[0]  # number of articles, number of features
    X_top_n = []
    for i in range(N):
        tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]
        top_n = feature_array[tfidf_sorting][:n]
        # if i%100==0:
        # print("\n", news_df.title[i])
        # print(top_n)

        X_top_n.append(" ".join(top_n))
    X_top_n = np.array(X_top_n).reshape((N,))
    # print(X_top_n.shape)
    # print(X_top_n[:5])

    # FastText Embedding 
    E = np.zeros((N, 120))
    for i in range(N):
        # E[i,:]=model.get_sentence_vector(X_top_n[i]) fasttext pckg version
        E[i, :] = model[X_top_n[i]]  # fine-tuned fasttext embedding using gensim

    return E


## Find closest article using cosine similarity

def find_closest(M1, M2, metric='cosine'):
    Dist = cdist(M1.reshape(1, -1), M2, metric=metric)
    # print(Dist.shape)
    i_closest = np.argmin(Dist)
    # print("\nCosine similarity : %1.3f" %(1-np.ravel(Dist)[np.argmin(Dist)]))
    return i_closest


### Reco pipeline ###

start = time.time()

print("\n\n\n>>> Test Reco / PCA / n = 4, 8, 12 <<<\n\n")

start = time.time()
reco_sets = [set() for k in range(len(sample_df))]

for n in [4, 8, 12]:
    # load embedding, standard scaler and PCA objects
    E_n_reduced = utils[n]["embedding"]
    Std = utils[n]["std_scaler"]
    pca = utils[n]["pca"]
    # réduction de E_sample
    E_sample_n = embed_top_n(sample_df, tfidf_vectorizer, n, model)
    E_sample_n_df = pd.DataFrame(E_sample_n)  # transformation en DataFrame
    E_sample_n_df = Std.transform(E_sample_n_df)  # Standard scaling avant PCA
    E_sample_n_reduced = pca.transform(E_sample_n_df)
    # back to numpy arrays    
    E_sample_n_reduced = np.array(E_sample_n_reduced)
    # Reco :  Sample articles + Archive articles
    for i0 in range(len(sample_df)):
        i_closest = find_closest(E_sample_n_reduced[i0, :], E_n_reduced)
        reco_sets[i0].add(i_closest)
print("\n>>> Intermediary <<< Elapsed: ", (time.time() - start))

# Take a look at the results
for i0 in range(len(sample_df)):
    print("\n\n    >>> Input Article : ", sample_df.title[i0])
    print(sample_df.text[i0][:500])
    for i_closest in list(reco_sets[i0]):
        print("\n Reco --- ", lemonde_df.title[i_closest])
        print(lemonde_df.text[i_closest][:500])

print("Elapsed: ", (time.time() - start))

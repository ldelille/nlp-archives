#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:15:58 2021

@author: eva
"""

import pickle 
import numpy as np 

from spacy.lang.fr import French
parser = French()
from nltk.stem.snowball import FrenchStemmer 
stemmer = FrenchStemmer()

import re 
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from random import choices

# Kmeans and corresponding tfidf

with open('tfidf_vectorizer_vocab_LOC', 'rb') as file:
    tfidf_vectorizer_vocab_LOC = pickle.load(file)

with open('kmeans_vocab_LOC', 'rb') as file:
    kmeans_vocab_LOC = pickle.load(file)

with open('tfidf_vectorizer_base', 'rb') as file:
    tfidf_vectorizer_base = pickle.load(file)

with open('tfidf_vectorizer_vocab', 'rb') as file:
    tfidf_vectorizer_vocab = pickle.load(file)

# Load sklearn tfidf pipelines 

with open('best_topic_lr_basic_vocab', 'rb') as file:
    best_topic_lr_basic_vocab = pickle.load(file)

with open('best_topic_lr_basic_vocab_l2', 'rb') as file:
    best_topic_lr_basic_vocab_l2 = pickle.load(file)
    
with open('best_topic_rf_basic_vocab', 'rb') as file:
    best_topic_rf_basic_vocab = pickle.load(file)

with open('best_geo_rf_entity_vocab', 'rb') as file:
    best_geo_rf_entity_vocab = pickle.load(file)

    
# topic codes

with open('geo_code_dic', 'rb') as file:
    geo_code_dic = pickle.load(file)
    
with open('topic_code_dic', 'rb') as file:
    topic_code_dic = pickle.load(file)
    
print(geo_code_dic)
print(topic_code_dic)

target_geo = ['afrique','asie', 'europe', 'france', 'am. latine', 'moyen-orient', 'espace', 'usa', 'monde']
target_topic = ['culture', 'economie', 'judiciaire', 'divers',  'politique', 'sciences', 'societe', 'sport']

## Functions -- Preprocessing -- ##

# preprocessing for the sample articles
my_fr_stop = {'celui-ci', 'de', 'serait', 'fussions', 'aux', 'seriez', 'c', 'notre', 'une',
              'seraient', 'avons', 'eûmes', 'ayantes', 'sur', 'alors', 'ayons', 'ait', 'du', 
              'ils', 'serai', 'fussent', 'vingt', 'avais', 'eus', 'eurent', 'm', 'ne', 'fûtes', 
              'sept', 'neuf', 'fusses', 'la', 'eussiez', 'vos', 'eût', 'te', 'huit', 'aura', 
              'j', 'étante', 'eut', 'ont', 'elle', 'leur', 'avec', 'ni', 'mais', 'quatre', 
              'aviez', 'sera', 'tes', 'ayants', 'à', 'soient', 'votre', 'ou', 'y', 'et', 
              'avions', 'je', 'fusse', 'fût', 'fut', 'd', 'ma', 'ce', 'il', 'étaient', 
              'par', 'ton', 'donc', 'soyons', 'les', 'sommes', 'ayante', 'dix', 'treize', 
              'comme', 'aient', 'seront', 'aie', 'eûtes', 'que', 'étais', 'ayez', 'auriez', 
              'seras', 'eux', 'n', 'vous', 'eussions', 'fûmes', 'ayant', 'deux', 'étiez', 
              'aurait', 'ai', 'auraient', 'cinquante', 'étions', 'cette', 'avoir', 't', 
              'nous', 'des', 'cent', 'toi', 'quinze', 'qui', 'sont', 'or', 'suis', 'serons',
              'eue', 'au', 'cinq', 'seize', 'quarante', 'me', 'où', 'pour', 'car', 'étant', 
              'depuis', 'était', 'l', 's', 'même', 'es', 'avait', 'sois', 'le', 'avaient', 
              'son', 'étés', 'soixante', 'onze', 'être', 'avez', 'aurais', 'auras', 'eusses', 
              'nos', 'étées', 'étantes', 'furent', 'ainsi', 'ses', 'six', 'autre', 'eues', 'été',
              'est', 'aies', 'se', 'celui-là', 'faire', 'soit', 'aurez', 'pas', 'en', 'moi', 
              'sa', 'on', 'auront', 'êtes', 'trente', 'quoi', 'quatorze', 'douze', 'étée', 'mes',
              'serions', 'fussiez', 'après', 'cet', 'mon', 'ta', 'trois', 'soyez', 'étants', 
              'dans', 'qu', 'as', 'celui', 'eu', 'quand', 'serez', 'lui', 'aurons', 'eusse', 
              'ces', 'tu', 'eussent', 'fus', 'avant', 'aurions', 'aurai', 'serais', 'un'}

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif '@' in str(token):
            lda_tokens+=str(token).split('@')
        else:
            lda_tokens.append(token.lower_)
    return [t for t in lda_tokens if len(str(t))>0]

def prepare_text_stem(text):
    """
    Input:
    ------
    text: string, raw text
    
    Output:
    ------
    tokens: list of string, tokenized, filtered and lemmatized words from the input text
    """
    tokens = tokenize(text) # split and lower case
    tokens=[re.sub(r'\b\d+\b', '', token).strip(' ') for token in tokens] # get rid of digits
    tokens = [token for token in tokens if len(token) > 3] # arbitrary length, +get rid of empty strings
    tokens = [token for token in tokens if token not in my_fr_stop] # stopwords
    #print("Remaining tokens : ", tokens)
    tokens = [stemmer.stem(token) for token in tokens] # obtain lemmas
    return tokens  

## Similarity ##
    
def most_similar(sample_articles, archive_articles, transformer=tfidf_vectorizer_base, refit = True):
    """
    Returns the index of the most similar articles
    sample_articles : dataframe, must have "text" col
    archive_articles : idem
    """
    if refit == True : 
        X = transformer.fit_transform(archive_articles.clean_text) # in case not already fitted, easier for testing
    else :
        X = transformer.transform(archive_articles.clean_text)
    sample_X = transformer.transform(sample_articles.clean_text)
    # compute similarity matrix
    sim_matrix = np.array([[cosine_similarity(sample_X[i], X[j]) for i in range(sample_X.shape[0])] for j in range(X.shape[0])])
    sim_matrix=sim_matrix.reshape([sim_matrix.shape[0], sim_matrix.shape[1]])
    reco_matrix = np.argmax(sim_matrix, axis=0)
    print(np.max(sim_matrix, axis=0))
    return reco_matrix

## Filtering ##
    
# note : not optimized, greedy filtering just for the sake of the demo
def filter_topic(sample_article, archives_articles):
    my_geo_code = sample_article.geo_code.iloc[0]
    my_topic_code = sample_article.topic_code.iloc[0]
    and_df = archives_articles[(archives_articles["geo_code"]==my_geo_code)&(archives_articles["topic_code"]==my_topic_code)].reset_index()
    if len(and_df)>0:
        ans = and_df
        print("# candidates:", len (ans), "(AND)")

    else :
        ans = archives_articles[(archives_articles["geo_code"]==my_geo_code)|(archives_articles["topic_code"]==my_topic_code)].reset_index()
        print("# candidates:", len (ans), "(OR)")
    return ans

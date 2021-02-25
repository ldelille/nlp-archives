#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:07:39 2021

@author: eva
"""
### Imports ###
import fasttext as ft

from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np
import pickle
import time
from joblib import load

import spacy
from nltk.stem.snowball import FrenchStemmer

from spacy.lang.fr import French

try:
    from nltk.corpus import stopwords
except:
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords

import re


class RecoArticle:
    def __init__(self):
        self.fr_stop = set(stopwords.words('french'))
        self.my_fr_stop = self.fr_stop.union({'ce', 'celui', 'cette', 'cet', 'celui-là', 'celui-ci',
                                              'le', 'la', 'les', 'de', 'des', 'du',
                                              'mais', 'où', 'et', 'donc', 'or', 'ni', 'car', 'depuis', 'quand', 'que',
                                              'qui',
                                              'quoi',
                                              'ainsi', 'alors', 'avant', 'après', 'comme',
                                              'être', 'avoir', 'faire',
                                              'autre'},
                                             {'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
                                              'dix',
                                              'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize',
                                              'vingt', 'trente', 'quarante', 'cinquante', 'soixante', 'cent'})
        self.parser = French()
        self.stemmer = FrenchStemmer()
        self.utils = dict()
        self.embed_list = []
        self.lemonde_df = pd.read_csv("../lemonde_ready.csv")

    def tokenize(self, text):
        lda_tokens = []
        tokens = self.parser(text)
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

    def prepare_text_stem(self, text):
        """
        Input:
        ------
        text: string, raw text

        Output:
        ------
        tokens: list of string, tokenized, filtered and lemmatized words from the input text
        """
        tokens = self.tokenize(text)  # split and lower case
        tokens = [re.sub(r'\b\d+\b', '', token).strip(' ') for token in tokens]  # get rid of digits
        tokens = [token for token in tokens if len(token) > 3]  # arbitrary length, +get rid of empty strings
        tokens = [token for token in tokens if token not in self.my_fr_stop]  # stopwords
        tokens = [self.stemmer.stem(token) for token in tokens]  # obtain lemmas
        return tokens

    def preprocess(self, news_df):
        """
        news_df must be a dataframe with text and title columns
        """
        title_tokens = []
        text_tokens = []

        ## Apply on titles ##
        for t in news_df['title']:
            tokens = self.prepare_text_stem(t)
            title_tokens.append(tokens)

        ## Apply on bodies ##

        for t in news_df.text:
            tokens = self.prepare_text_stem(t)
            text_tokens.append(tokens)

        # save the preprocessed text
        news_df["clean_text"] = [' '.join(text_tokens[i]) for i in range(len(text_tokens))]
        news_df["clean_title"] = [' '.join(title_tokens[i]) for i in range(len(title_tokens))]

        return news_df

    def embed_top_n(self, news_df, tfidf_vectorizer, n, model):
        """
        Input
        -----
        news_df : Pandas Dataframe, must contain a "clean_text" column.
        tfidf_vectorizer : already fitted tfidf vectorizer
        n : number of top tfidf words we want to keep for the embedding
        model : fasttext pre-trained and preloaded model. Hard-coded here : 300-dimensional embedding.

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
        E = np.zeros((N, 300))
        for i in range(N):
            E[i, :] = model.get_sentence_vector(X_top_n[i])

        return E

    def find_closest(self, M1, M2, metric='cosine'):
        Dist = cdist(M1.reshape(1, -1), M2, metric=metric)
        # print(Dist.shape)
        i_closest = np.argmin(Dist)
        # print("\nCosine similarity : %1.3f" %(1-np.ravel(Dist)[np.argmin(Dist)]))
        return i_closest

    def load_models(self):
        spacy.load('fr_core_news_sm')

        nlp = spacy.load("fr_core_news_sm")
        model = ft.load_model('../pipelines/cc.fr.300.bin')  # Pré-requis : installation de fasttext / cc.fr.300.bin
        sample_df = pd.read_excel("../sample_articles.xlsx")  # one level above current folder
        with open('../tfidf_vectorizer_base', 'rb') as handle:  # tfidf model
            tfidf_vectorizer = pickle.load(handle)
        for n in [4, 8, 12]:
            self.utils[n] = dict()
            self.utils[n]["embedding"] = np.loadtxt("../E_" + str(n) + "_leMonde_reduced.txt")
            self.utils[n]["std_scaler"] = load("../std_scaler_" + str(n) + ".joblib")
            self.utils[n]["pca"] = load("../pca_" + str(n) + ".joblib")

        sample_df = self.preprocess(sample_df)

        for n in [4, 8, 12]:
            Std = self.utils[n]["std_scaler"]
            pca = self.utils[n]["pca"]
            # réduction de E_sample
            E_sample_n = self.embed_top_n(sample_df, tfidf_vectorizer, n, model)
            E_sample_n_df = pd.DataFrame(E_sample_n)  # transformation en DataFrame
            E_sample_n_df = Std.transform(E_sample_n_df)  # Standard scaling avant PCA
            E_sample_n_reduced = pca.transform(E_sample_n_df)
            # back to numpy arrays
            E_sample_n_reduced = np.array(E_sample_n_reduced)
            self.embed_list.append(E_sample_n_reduced)
            # Reco :  Sample articles + Archive articles

    def launch_reco(self, article_id, only_titles_needed=True):
        reco_set = set()
        titles_reco = []
        texts_reco = []
        for embed, n in zip(self.embed_list, [4, 8, 12]):
            i_closest = self.find_closest(embed[article_id, :], self.utils[n]["embedding"])
            reco_set.add(i_closest)
        for i_closest in reco_set:
            titles_reco.append(self.lemonde_df.title[i_closest])
            texts_reco.append(self.lemonde_df.text[i_closest][:])
        if only_titles_needed:
            return titles_reco
        else:
            return titles_reco, texts_reco


# if __name__ == '__main__':
#     test_article = RecoArticle()
#     test_article.load_models()
#     print(test_article.launch_reco(3))
#     print(test_article.launch_reco(2))


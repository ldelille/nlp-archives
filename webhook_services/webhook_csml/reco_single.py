#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:07:39 2021

@author: eva
"""
### Imports ###

import pickle
import random

import numpy as np
import pandas as pd
import spacy
from gensim.models.fasttext import FastText
from joblib import load
from nltk.stem.snowball import FrenchStemmer
from scipy.spatial.distance import cdist
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
        self.embed_list_from_parsing = []
        self.lemonde_df = pd.read_csv("../../recommendation_models/archives_ready.csv")
        self.tfidf_vectorizer = None
        self.model = None
        self.sample_df = None
        self.embed_list_from_keywords = []
        self.warn = []
        self.keyword_df = None

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

    def prepare_text_stem(self, text, is_keyword=False):
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
        if not is_keyword:
            tokens = [token for token in tokens if len(token) > 3]  # arbitrary length, +get rid of empty strings
        tokens = [token for token in tokens if token not in self.my_fr_stop]  # stopwords
        tokens = [self.stemmer.stem(token) for token in tokens]  # obtain lemmas
        return tokens

    def preprocess(self, news_df, is_keyword=False):
        """
        news_df must be a dataframe with text and title columns
        """
        title_tokens = []
        text_tokens = []

        ## Apply on titles ##
        for t in news_df['title']:
            tokens = self.prepare_text_stem(t, is_keyword)
            title_tokens.append(tokens)

        ## Apply on bodies ##

        for t in news_df.text:
            tokens = self.prepare_text_stem(t, is_keyword)
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
        E = np.zeros((N, 120))
        for i in range(N):
            E[i, :] = model[X_top_n[i]]

        return E

    def embed_top_n_keywords(self, news_df, tfidf_vectorizer, n, model):
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

        # Handle no matching keyword scenario
        empty_df = pd.DataFrame({"title": [''], 'text': [''], 'year': [2021], 'clean_text': [''], 'clean_title': ['']})
        X_empty = tfidf_vectorizer.transform(empty_df.clean_text)  # X is a matrix
        # get features of tfidf
        feature_array = np.array(tfidf_vectorizer.get_feature_names())
        # extract top n words
        tfidf_sorting_empty = np.argsort(X_empty[0].toarray()).flatten()[::-1]
        top_n_empty = feature_array[tfidf_sorting_empty][:n]
        print("Top n words for empty request: ", top_n_empty)

        # Apply tfidf to the dataset
        X = tfidf_vectorizer.transform(news_df.clean_text)  # X is a matrix
        print("Shape of X : ", X.shape)
        # get features of tfidf
        feature_array = np.array(tfidf_vectorizer.get_feature_names())
        # extract top n words
        N = X.shape[0]  # number of articles, number of features
        X_top_n = []
        warn = []
        for i in range(N):
            tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]
            # print("Shape of sorted vectors: ", tfidf_sorting.shape)
            top_n = feature_array[tfidf_sorting][:n]
            if list(top_n) != list(top_n_empty):
                print("Top n words : ", top_n)
                warn.append(False)
            else:
                print("Warning - try to elaborate your request")
                warn.append(True)

            X_top_n.append(" ".join(top_n))
        X_top_n = np.array(X_top_n).reshape((N,))
        print(X_top_n)

        # FastText Embedding
        E = np.zeros((N, 120))
        for i in range(N):
            # E[i,:]=model.get_sentence_vector(X_top_n[i]) fasttext pckg version
            E[i, :] = model[X_top_n[i]]  # fine-tuned fasttext embedding using gensim

        return E, warn

    def find_closest(self, M1, M2, metric='cosine'):
        Dist = cdist(M1.reshape(1, -1), M2, metric=metric)
        i_closest = np.argmin(Dist)
        return i_closest

    def find_closest_wDates(self, M1, M2, mask, metric='cosine'):
        Dist = cdist(M1.reshape(1, -1), M2, metric=metric)
        mask = mask.reshape(Dist.shape[0], -1)
        Dist = mask + Dist
        i_closest = np.argmin(Dist)
        sim = 1 - np.ravel(Dist)[np.argmin(Dist)]
        print("\nCosine similarity : %1.3f" % sim)
        if sim < 0.75:
            print(
                "Changer les contraintes temporelles ou modifiez votre requête pour obtenir des articles plus similaires.")
        return i_closest

    def load_models(self):
        spacy.load('fr_core_news_sm')
        self.model = FastText.load("../../recommendation_models/fasttext_finetuned_50K_120d.model")
        with open('../../recommendation_models/tfidf_vectorizer_50K', 'rb') as handle:  # tfidf model
            self.tfidf_vectorizer = pickle.load(handle)
        for n in [4, 8, 12]:
            self.utils[n] = dict()
            self.utils[n]["embedding"] = np.loadtxt(
                "../../recommendation_models/./E_" + str(n) + "_archives_reduced.txt")
            self.utils[n]["std_scaler"] = load("../../recommendation_models/std_scaler_50K_" + str(n) + ".joblib")
            self.utils[n]["pca"] = load("../../recommendation_models/pca_50K_" + str(n) + ".joblib")

    def compute_embeddings_from_sample(self):
        sample_df = pd.read_excel("../../input_articles/recommendation/sample_articles.xlsx")
        self.sample_df = self.preprocess(sample_df)
        for n in [4, 8, 12]:
            Std = self.utils[n]["std_scaler"]
            pca = self.utils[n]["pca"]
            # réduction de E_sample
            E_sample_n = self.embed_top_n(self.sample_df, self.tfidf_vectorizer, n, self.model)
            E_sample_n_df = pd.DataFrame(E_sample_n)  # transformation en DataFrame
            E_sample_n_df = Std.transform(E_sample_n_df)  # Standard scaling avant PCA
            E_sample_n_reduced = pca.transform(E_sample_n_df)
            # back to numpy arrays
            E_sample_n_reduced = np.array(E_sample_n_reduced)
            self.embed_list.append(E_sample_n_reduced)
            # Reco :  Sample articles + Archive articles

    def compute_embeddings_from_parsed_article(self, data):
        article_df = pd.DataFrame(
            {'title': data['title'], 'text': data['text'], 'url': data['url'], 'year': data['date_published']},
            index=[0])
        article_df = self.preprocess(article_df)
        self.embed_list_from_parsing = []
        for n in [4, 8, 12]:
            Std = self.utils[n]["std_scaler"]
            pca = self.utils[n]["pca"]
            # réduction de E_sample
            E_sample_n = self.embed_top_n(article_df, self.tfidf_vectorizer, n, self.model)
            E_sample_n_df = pd.DataFrame(E_sample_n)  # transformation en DataFrame
            E_sample_n_df = Std.transform(E_sample_n_df)  # Standard scaling avant PCA
            E_sample_n_reduced = pca.transform(E_sample_n_df)
            # back to numpy arrays
            E_sample_n_reduced = np.array(E_sample_n_reduced)
            self.embed_list_from_parsing.append(E_sample_n_reduced)

    def compute_embeddings_from_keywords(self, keys_data):
        self.keyword_df = pd.DataFrame({'title': '', 'text': keys_data['data'],
                                        'year_min': keys_data['year_min'], 'year_max': keys_data['year_max']})
        self.keyword_df = self.preprocess(self.keyword_df, is_keyword=True)
        self.embed_list_from_keywords = []
        result = []
        warn = True
        for n in [4, 8, 12]:
            # load embedding, standard scaler and PCA objects
            Std = self.utils[n]["std_scaler"]
            pca = self.utils[n]["pca"]
            # réduction de E_sample
            E_sample_n, self.warn_n = self.embed_top_n_keywords(self.keyword_df, self.tfidf_vectorizer, n, self.model)
            print('self.warn_n', self.warn_n)
            E_sample_n_df = pd.DataFrame(E_sample_n)  # transformation en DataFrame
            E_sample_n_df = Std.transform(E_sample_n_df)  # Standard scaling avant PCA
            E_sample_n_reduced = pca.transform(E_sample_n_df)
            # back to numpy arrays
            self.embed_list_from_keywords.append(E_sample_n_reduced)
            # Reco :  Sample articles + Archive articles
            # mask for filtering by dates:
            y_min, y_max = int(self.keyword_df.year_min[0]), int(self.keyword_df.year_max[0])
            print("Contraintes temporelles : article daté entre %s et %s." % (y_min, y_max))
            mask = np.array(self.lemonde_df['year'])
            mask = 1 * (mask >= y_min) * (mask <= y_max)
            if np.sum(mask) > 0:
                print("Des articles d'archive correspondent aux contraintes temporelles fournies")
                if self.warn_n[0]:
                    result.append(random.randrange(len(self.utils[n]["embedding"])))
                else:
                    i_closest = self.find_closest_wDates(E_sample_n_reduced[0, :], self.utils[n]["embedding"], mask)
                    result.append(i_closest)
            else:
                print(
                    "Les bornes temporelles fournies ne permettent pas d'obtenir de résultats. Annulation des contraintes.")
                if self.warn_n[0] == True:
                    result.append(random.randrange(len(self.utils[n]["embedding"])))
                else:
                    i_closest = self.find_closest(E_sample_n_reduced[0, :], self.utils[n]["embedding"])
                    result.append(i_closest)
        if self.warn_n[0]:
            return {"result": "reco did not succeed"}
        else:
            res = {}
            res["result"] = {}
            print('result', result)
            for cpt, article in enumerate(result):
                res["result"]["article_" + str(cpt)] = {}
                res["result"]["article_" + str(cpt)]["title"] = self.lemonde_df.title[article][:]
                # res["result"]["article_" + str(cpt)]["text"] = self.lemonde_df.text[article][:]
                res["result"]["article_" + str(cpt)]["url"] = self.lemonde_df.url[article][:]
            return res

    def launch_reco_from_id(self, article_id):
        reco_list = []
        res = {}
        res["input_article"] = {}
        res["input_article"]["title"] = self.sample_df.title[article_id]
        res["input_article"]["url"] = self.sample_df.url[article_id]
        res["input_article"]["date_published"] = str(self.sample_df.year[article_id])
        for embed, n in zip(self.embed_list, [4, 8, 12]):
            i_closest = self.find_closest(embed[article_id, :], self.utils[n]["embedding"])
            reco_list.append(i_closest)
        res["result"] = {}
        res["result"]["article_count"] = len(reco_list)
        for cpt, i_closest in enumerate(reco_list):
            res["result"]["article_" + str(cpt)] = {}
            # res["result"]["article_"+ str(cpt)]["text"] = self.lemonde_df.text[i_closest][:]
            res["result"]["article_" + str(cpt)]["title"] = self.lemonde_df.title[i_closest][:]
            res["result"]["article_" + str(cpt)]["date_published"] = str(self.lemonde_df.date_published[i_closest][:])
            res["result"]["article_" + str(cpt)]["url"] = self.lemonde_df.url[i_closest][:]
        return res

    def launch_reco_from_parsed_article(self):
        reco_list = []
        res = {}
        for embed, n in zip(self.embed_list_from_parsing, [4, 8, 12]):
            i_closest = self.find_closest(embed[0, :], self.utils[n]["embedding"])
            reco_list.append(i_closest)
        res["result"] = {}
        res["result"]["article_count"] = len(reco_list)
        for cpt, i_closest in enumerate(reco_list):
            res["result"]["article_" + str(cpt)] = {}
            res["result"]["article_" + str(cpt)]["text"] = self.lemonde_df.text[i_closest][:]
            res["result"]["article_" + str(cpt)]["title"] = self.lemonde_df.title[i_closest][:]
            res["result"]["article_" + str(cpt)]["date_published"] = self.lemonde_df.date_published[i_closest][:]
            res["result"]["article_" + str(cpt)]["url"] = self.lemonde_df.url[i_closest][:]
        return res

    def launch_reco_from_keyworsds(self):
        pass


if __name__ == '__main__':
    test_article = RecoArticle()
    test_article.load_models()
    test_article.compute_embeddings_from_sample()
    test_keywords = {
        "data": [
            "régions France parcs naturels"],
        "year_min": [1950],
        "year_max": [1950]
    }
    test_article.compute_embeddings_from_keywords(test_keywords)
    print(test_article.compute_embeddings_from_keywords(test_keywords))

    # print(test_article.launch_reco_from_id(10))

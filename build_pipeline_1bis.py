#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:05:28 2021

@author: eva
"""

####     Construction d'embeddings avec fasttext     ####
####   pour calculer la similarité entre documents   ####


import fasttext as ft
from scipy.spatial.distance import cdist # a row, + a matrix

import pandas as pd
import pickle
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from text_preprocessing import preprocess 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load

"""
## 1 - Fusion des Datasets Le Monde + Ouest France ##

df1 = pd.read_csv("./articles.csv") # Ouest France 
df2 = pd.read_csv("./lemonde.csv") # Le Monde
print(df1.columns)
print(df2.columns)
print(len(df1))
print(len(df2))

# full outer join to merge the dataframes - revient à un append
# on garde le nom "lemonde_df" pour réutiliser code 
# common cols are title, text, year and url
lemonde_df = pd.merge(df2, df1, left_on=['title', 'text', 'year', 'url'], right_on=['title', 'text', 'year', 'url'], how='outer') 
print(len(lemonde_df)==len(df1)+len(df2))
print(lemonde_df.columns)
lemonde_df.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1, inplace=True)

## 2 - Preprocessing et sauvegarde ##

start = time.time()
lemonde_df = preprocess(lemonde_df)
print("Elapsed: ", (time.time()-start)) 

print(lemonde_df.columns)
"""
lemonde_df = pd.read_csv("archives_ready.csv")

## 3 - Tfidf ##
"""
# fit tf-idf on all preprocessed texts with constraints on document frequency and number of tokens
tfidf_vectorizer = TfidfVectorizer(max_df = 20000, max_features=1000, smooth_idf=True,use_idf=True) #strip_accents=True
X = tfidf_vectorizer.fit_transform(lemonde_df.clean_text) # X is a matrix
# take a look at the result
print(tfidf_vectorizer.get_feature_names())
print(X.shape)
# save the tfidf matrix
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer_50K', 'wb'))
"""
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer_50K', 'rb'))

## 4 - Fine tuning de fasttext sur le corpus ##

# TODO dans un deuxième temps 
from gensim.models.fasttext import FastText
model = FastText.load("fasttext_finetuned_50K_120d.model") 


## 5 - Construction d'embeddings à partir des mots saillants de chaque article ##
##     n = 4, 8, 12 mots

#model = ft.load_model('cc.fr.300.bin') # Loading model for French

### Apply tfidf + FastText embedding on top n words ###

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
    X = tfidf_vectorizer.transform(news_df.clean_text) # X is a matrix
    #print(X.shape)
    # get features of tfidf
    feature_array = np.array(tfidf_vectorizer.get_feature_names())
    
    # extract top n words
    N = X.shape[0] # number of articles, number of features 
    X_top_n = []
    for i in range(N):
        tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]
        top_n = feature_array[tfidf_sorting][:n]
        #if i%100==0:
            #print("\n", news_df.title[i])
            #print(top_n)
            
        X_top_n.append(" ".join(top_n))        
    X_top_n = np.array(X_top_n).reshape((N,))
    #print(X_top_n.shape)
    #print(X_top_n[:5])
    
    # FastText Embedding 
    E=np.zeros((N,120)) # instead of 300
    for i in range(N):
        #E[i,:]=model.get_sentence_vector(X_top_n[i]) from fasttext package - out of the box
        E[i,:]=model[X_top_n[i]] # fine-tuned fasttext embedding using gensim 
    
    return E

def find_closest(M1, M2, metric='cosine'):
    print(M1.reshape(1, -1).shape)
    print(M2.shape)
    Dist = cdist(M1.reshape(1, -1), M2, metric=metric)
    #print(Dist.shape)
    i_closest = np.argmin(Dist)
    #print("\nCosine similarity : %1.3f" %(1-np.ravel(Dist)[np.argmin(Dist)]))
    return i_closest


# Sauver embedding de taille réduite
# Objectif : conserver 80% de variance expliquée

start = time.time()  

for n in [4, 8, 12]:
    E_n = embed_top_n(lemonde_df, tfidf_vectorizer, n, model) #np.loadtxt("./E_"+str(n)+"leMonde.txt") # chargement de la matrice numpy
    E_n_df = pd.DataFrame(E_n) # transformation en DataFrame
    Std = StandardScaler().fit(E_n_df)
    E_n_df = Std.transform(E_n_df) # Standard scaling avant PCA    
    pca = PCA(n_components=0.8, svd_solver = 'full')
    E_n_reduced = pca.fit_transform(E_n_df)
    print("Number of principal components : ", E_n_reduced.shape[1])    
    # back to numpy arrays
    E_n_reduced = np.array(E_n_reduced)
    # save objects
    np.savetxt("./E_"+str(n)+"_archives_reduced.txt", E_n_reduced)
    # dump standard scaler and PCA objects
    dump(Std, "std_scaler_50K_"+str(n)+".joblib")
    dump(pca, "pca_50K_"+str(n)+".joblib")

print("Elapsed: ", (time.time()-start)) 


## 6 - Tester le système de recommandation par similarité cosinus ##

sample_df=pd.read_excel("./sample_articles.xlsx")
sample_df = preprocess(sample_df)

start = time.time()
# preload reduced embeddings, standard scaler and PCA objects
utils = dict()
for n in [4, 8, 12]:
    utils[n]=dict()
    utils[n]["embedding"] = np.loadtxt("./E_"+str(n)+"_archives_reduced.txt") 
    utils[n]["std_scaler"] = load("std_scaler_50K_"+str(n)+".joblib")
    utils[n]["pca"] =  load("pca_50K_"+str(n)+".joblib")     
print("\n>>> Pre-loading  <<< Elapsed: ", (time.time()-start)) 
  

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
    E_sample_n_df = pd.DataFrame(E_sample_n) # transformation en DataFrame
    E_sample_n_df = Std.transform(E_sample_n_df) # Standard scaling avant PCA
    E_sample_n_reduced = pca.transform(E_sample_n_df)
    # back to numpy arrays    
    E_sample_n_reduced = np.array(E_sample_n_reduced)
    # Reco :  Sample articles + Archive articles
    for i0 in range(len(sample_df)):
        i_closest = find_closest(E_sample_n_reduced[i0,:], E_n_reduced)   
        reco_sets[i0].add(i_closest)
print("\n>>> Intermediary <<< Elapsed: ", (time.time()-start)) 

# Take a look at the results
for i0 in range(len(sample_df)):
    print("\n\n    >>> Input Article : ", sample_df.title[i0])
    print(sample_df.text[i0][:500])
    for i_closest in list(reco_sets[i0]):
        print("\n Reco --- ", lemonde_df.title[i_closest])
        print(lemonde_df.text[i_closest][:500])

print("Elapsed: ", (time.time()-start)) 

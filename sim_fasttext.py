#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:07:39 2021

@author: eva
"""

import fasttext as ft

#from scipy.spatial.distance import cosine # a row + another row
from scipy.spatial.distance import cdist # a row, + a matrix
#from scipy.spatial.distance import pdist # pairwise, all rows  all rows

import pandas as pd
import pickle
import numpy as np

import time
from text_preprocessing import preprocess 

### Load data and models ###

model = ft.load_model('cc.fr.300.bin') # Loading model for French

news_df = pd.read_csv("./articles.csv") # 726 news articles
sample_df=pd.read_excel("sample_articles.xlsx")

with open('tfidf_vectorizer_base', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)

# tfidf model 

### Preprocessing ###

news_df = preprocess(news_df)
sample_df = preprocess(sample_df)
### Apply tfidf + FastText embedding on top n words ###

def embed_top_n(news_df, tfidf_vectorizer, n, model):
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
    E=np.zeros((N,300))
    for i in range(N):
        E[i,:]=model.get_sentence_vector(X_top_n[i])
    
    return E

start = time.time()
#test
n=4
E = embed_top_n(news_df, tfidf_vectorizer, n, model)
E_sample = embed_top_n(sample_df, tfidf_vectorizer, n, model)

### Find the corpus article most similar to one particular article ###

def find_closest(M1, M2, metric='cosine'):
    Dist = cdist(M1.reshape(1, -1), M2, metric=metric)
    #print(Dist.shape)
    i_closest = np.argmin(Dist)
    #print("\nCosine similarity : %1.3f" %(1-np.ravel(Dist)[np.argmin(Dist)]))
    return i_closest

# test 0
i_closest = find_closest(E[0,:], E[1:, :])   
# Take a look at the results 
print("\n", news_df.title[0])
print(news_df.text[0][:500])
print("\n", news_df.title[i_closest+1])
print(news_df.text[i_closest+1][:500])

#print(X_top_n[0])
#print(X_top_n[i_closest+1])

# more tests (10 articles, archives) 
for i0 in range(10):
    i_closest = find_closest(E[i0,:], E[i0+1:, :])   
    # Take a look at the results 
    print("\n", news_df.title[i0])
    print(news_df.text[i0][:500])
    print("\n", news_df.title[i_closest+i0+1])
    print(news_df.text[i_closest+i0+1][:500])
    

# Sample articles + Archive articles
for i0 in range(len(sample_df)):
    i_closest = find_closest(E_sample[i0,:], E)   
    # Take a look at the results 
    print("\n", sample_df.title[i0])
    print(sample_df.text[i0][:500])
    print("\n", news_df.title[i_closest])
    print(news_df.text[i_closest][:500])

print("Elapsed: ", (time.time()-start))    
# test : find the most similar article pairs 
"""
NB : pdist returns a condensed distance matrix Y. 
For each  and  (where ),where m is the number of original observations. 
The metric dist(u=X[i], v=X[j]) is computed 
and stored in entry m * i + j - ((i + 2) * (i + 1)) // 2.
"""
#PDist = pdist(E, metric='cosine')
#print(PDist.shape)


### Tests sur articles du Monde ###

lemonde_df = pd.read_csv("./lemonde.csv")

start = time.time()
lemonde_df = preprocess(lemonde_df)
print("Elapsed: ", (time.time()-start)) 


start = time.time()   
n=16
E = embed_top_n(lemonde_df, tfidf_vectorizer, n, model)
E_sample = embed_top_n(sample_df, tfidf_vectorizer, n, model)

# Sample articles + Archive articles
for i0 in range(len(sample_df)):
    i_closest = find_closest(E_sample[i0,:], E)   
    # Take a look at the results 
    print("\n", sample_df.title[i0])
    print(sample_df.text[i0][:500])
    print("\n", lemonde_df.title[i_closest])
    print(lemonde_df.text[i_closest][:500])
print("Elapsed: ", (time.time()-start))   



### Get several recommandations for the same text ###
start = time.time()  

n_list = [4, 8, 12]
reco_sets = [set() for k in range(len(sample_df))] 
# element i is a set which elements are indices of archive articles
# element j of tuple i corresponds to a reco formulated with n_list[j] as constraint
 
for n in n_list : 
    #E = embed_top_n(lemonde_df, tfidf_vectorizer, n, model)
    E_sample = embed_top_n(sample_df, tfidf_vectorizer, n, model)
    fname = "E_"+str(n)+"leMonde.txt"
    E = np.loadtxt(fname)
    # Sample articles + Archive articles
    for i0 in range(len(sample_df)):
        i_closest = find_closest(E_sample[i0,:], E)   
        reco_sets[i0].add(i_closest)

print(reco_sets)
# Take a look at the results
for i0 in range(len(sample_df)):
    print("\n\nInput Article : ", sample_df.title[i0])
    print(sample_df.text[i0][:500])
    for i_closest in list(reco_sets[i0]):
        print("\n Reco --- ", lemonde_df.title[i_closest])
        print(lemonde_df.text[i_closest][:500])

print("Elapsed: ", (time.time()-start)) 

# save with preprocessing
lemonde_df.to_csv("./lemonde_ready.csv")

### test - avec ce qui a été sauvé ###

# objectif : abaisser le run time !!!
print("\n\n\n>>> Test Reco <<<\n\n")
start = time.time()
n_list = [4, 8, 12]
reco_sets = [set() for k in range(len(sample_df))] 
for n in n_list : 
    #E = embed_top_n(lemonde_df, tfidf_vectorizer, n, model)
    E_sample = embed_top_n(sample_df, tfidf_vectorizer, n, model)
    fname = "E_"+str(n)+"leMonde.txt"
    E = np.loadtxt(fname)
    # Sample articles + Archive articles
    for i0 in range(len(sample_df)):
        i_closest = find_closest(E_sample[i0,:], E)   
        reco_sets[i0].add(i_closest)
print("\n>>> Intermediary <<< Elapsed: ", (time.time()-start)) 
#print(reco_sets)
# Take a look at the results

for i0 in range(len(sample_df)):
    print("\n\n    >>> Input Article : ", sample_df.title[i0])
    print(sample_df.text[i0][:500])
    for i_closest in list(reco_sets[i0]):
        print("\n Reco --- ", lemonde_df.title[i_closest])
        print(lemonde_df.text[i_closest][:500])

print("Elapsed: ", (time.time()-start)) 


### Réduction de dimension : matrices E_n ###

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load

# Sauver embedding de taille réduite
# Objectif : conserver 80% de variance expliquée
for n in [4, 8, 12]:
    E_n = np.loadtxt("./E_"+str(n)+"leMonde.txt") # chargement de la matrice numpy
    E_n_df = pd.DataFrame(E_n) # transformation en DataFrame
    Std = StandardScaler().fit(E_n_df)
    E_n_df = Std.transform(E_n_df) # Standard scaling avant PCA    
    pca = PCA(n_components=0.8, svd_solver = 'full')
    E_n_reduced = pca.fit_transform(E_n_df)
    print("Number of principal components : ", E_n_reduced.shape[1])    
    # back to numpy arrays
    E_n_reduced = np.array(E_n_reduced)
    # save objects
    np.savetxt("./E_"+str(n)+"_leMonde_reduced.txt", E_n_reduced)
    # dump standard scaler and PCA objects
    dump(Std, "std_scaler_"+str(n)+".joblib")
    dump(pca, "pca_"+str(n)+".joblib")
    
# visualisation

import matplotlib.pyplot as plt 
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_)) 
plt.plot(pca.explained_variance_ratio_+np.cumsum(pca.explained_variance_ratio_)[0]) 
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.legend(('Cumulated explained Variance', 'Individual explained Variance '),loc='center right') 
plt.title('Explained variance - First PCA componants')
plt.show()


start = time.time()
# preload reduced embeddings, standard scaler and PCA objects
utils = dict()
for n in [4, 8, 12]:
    utils[n]=dict()
    utils[n]["embedding"] = np.loadtxt("./E_"+str(n)+"_leMonde_reduced.txt") 
    utils[n]["std_scaler"] = load("std_scaler_"+str(n)+".joblib")
    utils[n]["pca"] =  load("pca_"+str(n)+".joblib")     
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

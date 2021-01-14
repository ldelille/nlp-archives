#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:18:29 2021

@author: eva
"""

## Utility functions ##

from reco_utils import *


## Packages ##

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report



## Data ##

labeled_df=pd.read_csv("labeled_articles_clean.csv")
print(labeled_df[:5])

sample_df=pd.read_excel("sample_articles.xlsx")
print(sample_df[:5])


## --- Tests --- ##


# -- Preprocessing -- #

title_tokens = []
text_tokens = []

for t in sample_df.title:
    tokens = prepare_text_stem(t)
    title_tokens.append(tokens)
        
for t in sample_df.text:
    tokens = prepare_text_stem(t)
    text_tokens.append(tokens)

print(sample_df.title[:1])        
print(title_tokens[:1])
print(sample_df.text[:1][:20])  
print(text_tokens[0][:10])

# save the preprocessed text
sample_df["clean_text"]=[' '.join(text_tokens[i]) for i in range(len(text_tokens))]
sample_df["clean_title"]=[' '.join(title_tokens[i]) for i in range(len(title_tokens))]

# -- Geo predictions -- #
pred_geo=best_geo_rf_entity_vocab.predict(sample_df.clean_text)
pred_geo_labels = [geo_code_dic[x] for x in pred_geo]
print(pred_geo_labels)
print(classification_report(sample_df.geo_code,  pred_geo))

# -- Topic predictions -- #
pred_topic = best_topic_lr_basic_vocab.predict(sample_df.clean_text)
pred_topic_labels = [topic_code_dic[x] for x in pred_topic]
print(pred_topic_labels)
print(classification_report(sample_df.topic_code,  pred_topic))

# -- Similarity v1 -- #

# tfidf_vectorizer_vocab
reco = most_similar(sample_df, labeled_df, transformer=tfidf_vectorizer_vocab)
for i in range(len(sample_df)):
    print("\n\n **", sample_df.title[i])
    print("\nMost similar article:", labeled_df.title[reco[i]])
    print(labeled_df.text[reco[i]][:300], '...')

# tfidf_vectorizer_base
reco = most_similar(sample_df, labeled_df, transformer=tfidf_vectorizer_base)
for i in range(len(sample_df)):
    print("\n\n **", sample_df.title[i])
    print("\nMost similar article:", labeled_df.title[reco[i]])
    print(labeled_df.text[reco[i]][:300], '...')

# -- Clsutering -- #    
# Random articles from the same cluster according to k-means model
X_LOC = tfidf_vectorizer_vocab_LOC.transform(labeled_df.clean_text)
X_sample_LOC = tfidf_vectorizer_vocab_LOC.transform(sample_df.clean_text)

pred_ref_clusters = kmeans_vocab_LOC.predict(X_LOC)
pred_sample_cluster = kmeans_vocab_LOC.predict(X_sample_LOC)

print(pred_sample_cluster)

for i in range(len(sample_df)):
    sample_article = sample_df[sample_df.index==i] # df
    print("\n\n **", sample_df.title[i])
    cluster_i = pred_sample_cluster[i]
    index = pred_ref_clusters==cluster_i
    filtered_df_i = labeled_df[index]
    reco_i = choices(filtered_df_i.index, k=3)
    print("Recommandations: ")
    for r in reco_i:
        print(filtered_df_i[filtered_df_i.index==r].title.iloc[0])
        
# -- With filtering -- #
for i in range(len(sample_df)):
    sample_article = sample_df[sample_df.index==i] # df
    print("\n\n **", sample_df.title[i])
    print(sample_article.geo_code.iloc[0])
    filtered_df = filter_topic(sample_article, labeled_df) # df with new index col
    reco = most_similar(sample_article, filtered_df, transformer=tfidf_vectorizer_base)
    print(reco)    
    print("\nMost similar article:", filtered_df[filtered_df.index==reco[0]].title.iloc[0])
    print(filtered_df.text[reco[0]][:300], '...')
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:07:58 2021

@author: eva
"""

### Imports ###

import spacy

from nltk.stem.snowball import FrenchStemmer # already something 
stemmer = FrenchStemmer()

spacy.load('fr')
from spacy.lang.fr import French
parser = French()


try : 
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
    tokens = tokenize(text) # split and lower case
    tokens=[re.sub(r'\b\d+\b', '', token).strip(' ') for token in tokens] # get rid of digits
    tokens = [token for token in tokens if len(token) > 3] # arbitrary length, +get rid of empty strings
    tokens = [token for token in tokens if token not in my_fr_stop] # stopwords
    #print("Remaining tokens : ", tokens)
    tokens = [stemmer.stem(token) for token in tokens] # obtain lemmas
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
    news_df["clean_text"]=[' '.join(text_tokens[i]) for i in range(len(text_tokens))]
    news_df["clean_title"]=[' '.join(title_tokens[i]) for i in range(len(title_tokens))]
    
    return news_df

## test ##
    





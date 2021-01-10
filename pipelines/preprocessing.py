import nltk
import spacy
from nltk.corpus import stopwords
from nltk import re
import pandas as pd

spacy.load('fr')
from spacy.lang.fr import French

parser = French()

fr_stop = set(nltk.corpus.stopwords.words('french'))
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


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


class ArticlePreprocessor:
    def __init__(self, article: str):
        self.article = article


def main():
    news_df = pd.read_csv("./articles.csv")

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

nlp = spacy.load("fr_core_news_sm")

def prepare_text(text):
    """
    Input:
    ------
    text: string, raw text

    Output:
    ------
    tokens: list of string, tokenized, filtered and lemmatized words from the input text
    """
    tokens = tokenize(text)  # split and lower case
    tokens = [re.sub(r'\b\d+\b', '', token) for token in tokens]  # get rid of digits
    tokens = [token for token in tokens if len(token) > 4]  # arbitrary length, +get rid of empty strings
    tokens = [token for token in tokens if token not in my_fr_stop]  # stopwords
    doc = nlp(' '.join(tokens))  # pave the wave for spacy lemmatizer
    tokens = [token.lemma_ for token in doc]  # obtain lemmas
    return tokens


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
    title_tokens = []
    text_tokens = []
    ## Apply on titles ##
    for t in news_df.title:
        tokens = prepare_text(t)
        title_tokens.append(tokens)

    ## Apply on titles ##

    for t in news_df.text:
        tokens = prepare_text(t)
        text_tokens.append(tokens)

    print(news_df.title[:1])
    print(title_tokens[:1])
    print(news_df.text[:1][:20])
    print(text_tokens[0][:10])

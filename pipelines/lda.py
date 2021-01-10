import pickle

from gensim import corpora


class LDA():
    def __init__(self, title_tokens, text_tokens):
        self.title_tokens = title_tokens
        self.text_tokens = text_tokens


def main():


    dic_title = corpora.Dictionary(title_tokens)
    dic_text = corpora.Dictionary(text_tokens)
    corpus_title = [dic_title.doc2bow(token) for token in title_tokens]
    corpus_text = [dic_text.doc2bow(token) for token in text_tokens]

    pickle.dump(corpus_title, open('corpus_title.pkl', 'wb'))
    dic_title.save('dic_title.gensim')
    pickle.dump(corpus_title, open('corpus_text.pkl', 'wb'))
    dic_text.save('dic_text.gensim')
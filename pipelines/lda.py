import pickle

from gensim import corpora
import pandas as pd
from pipelines.preprocessing import ArticlePreprocessor


class LDA():
    def __init__(self, preprocessor: ArticlePreprocessor):
        self.preprocessor = preprocessor
        self.dic_title
        self.dic_text
        self.corpus_title
        self.corpus_text

    def launchLDA(self):
        dic_title = corpora.Dictionary(self.preprocessor.title_tokens)
        dic_text = corpora.Dictionary(self.preprocessor.text_tokens)
        corpus_title = [dic_title.doc2bow(token) for token in self.preprocessor.title_tokens]
        corpus_text = [dic_text.doc2bow(token) for token in self.preprocessor.text_tokens]

        pickle.dump(corpus_title, open('corpus_title.pkl', 'wb'))
        dic_title.save('dic_title.gensim')
        pickle.dump(corpus_title, open('corpus_text.pkl', 'wb'))
        dic_text.save('dic_text.gensim')


def main():
    news_df = pd.read_csv("./articles.csv")
    article_preprocessor = ArticlePreprocessor(news_df)
    article_preprocessor.fully_preprocess()
    lda = LDA(article_preprocessor)
    lda.launchLDA()


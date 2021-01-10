import pickle

import gensim
from gensim import corpora
import pandas as pd
from pipelines.preprocessing import ArticlePreprocessor


class LDA():
    def __init__(self, pre_processor: ArticlePreprocessor, nums_topics :int):
        self.preprocessor = pre_processor
        self.dic_title
        self.dic_text
        self.corpus_title
        self.corpus_text
        self.nums_topics = nums_topics
        self.topics = []
        self.model

    def launchLDA(self):
        dic_title = corpora.Dictionary(self.pre_processor.title_tokens)
        dic_text = corpora.Dictionary(self.pre_processor.text_tokens)
        corpus_title = [dic_title.doc2bow(token) for token in self.pre_processor.title_tokens]
        corpus_text = [dic_text.doc2bow(token) for token in self.pre_processor.text_tokens]

        pickle.dump(corpus_title, open('corpus_title.pkl', 'wb'))
        dic_title.save('dic_title.gensim')
        pickle.dump(corpus_title, open('corpus_text.pkl', 'wb'))
        dic_text.save('dic_text.gensim')

    def get_topics(self):
        NUM_TOPICS = 10
        for (corpus, dictionary) in [(self.corpus_title, self.dic_title), (self.corpus_text, self.dic_text)]:
            print('\nTopics')
            self.model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
            self.model.save('model5.gensim')
            self.topics = self.model.print_topics(num_words=4)


def main():
    news_df = pd.read_csv("./articles.csv")
    article_preprocessor = ArticlePreprocessor(news_df)
    article_preprocessor.fully_preprocess()
    lda = LDA(article_preprocessor)
    lda.launchLDA()
    lda.get_topics()
    for topic in lda.topics:
        print(topic)


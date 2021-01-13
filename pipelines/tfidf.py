import pandas as pd
# __Tf-idf with lemmatized words__
from pipelines.preprocessing import ArticlePreprocessor

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

cv = CountVectorizer()


def main():
    news_df = pd.read_csv("./articles.csv")
    article_preprocessor = ArticlePreprocessor(news_df)
    article_preprocessor.fully_preprocess()
    news_df['pre_title'] = [' '.join(word_list_i) for word_list_i in
                            article_preprocessor.title_tokens]  # preprocessed titles
    news_df['pre_text'] = [' '.join(word_list_i) for word_list_i in
                           article_preprocessor.text_tokens]  # preprocessed article bodies

    word_count_vector = cv.fit_transform(news_df.pre_text)
    print(word_count_vector.shape)
    # tf-idf with all the words
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # print idf values
    idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    # sort ascending
    idf.sort_values(by=['idf_weights'])

    # fit tf-idf on all preprocessed texts with constraints on document frequency and number of tokens
    tfidf_vectorizer=TfidfVectorizer(max_df = 700, max_features=500, smooth_idf=True,use_idf=True) #strip_accents=True
    X = tfidf_vectorizer.fit_transform(news_df.pre_text) # X is a matrix
    print(tfidf_vectorizer.get_feature_names())
    print(X.shape)

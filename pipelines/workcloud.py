from wordcloud import WordCloud

import pandas as pd

from pipelines.preprocessing import ArticlePreprocessor

#We can control the quality of our preprocessing by plotting a wordcloud.
def main():
    news_df = pd.read_csv("../input_articles/articles.csv")
    article_preprocessor = ArticlePreprocessor(news_df)
    article_preprocessor.fully_preprocess()
    # Join the different processed titles together.
    long_string = ','.join(list([' '.join(word_list_i) for word_list_i in article_preprocessor.title_tokens]))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()
    wordcloud.to_file('titleCloud.png')

if __name__ == '__main__':
    main()

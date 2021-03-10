# NewsBacklight

This NLP project aims at shedding a new light on present events by analysing newspaper articles from the past.

We develop a chatbot that provides newspaper readers with archive articles recommendations based on semantic similarities and topic recognition.


### Contents of this repository

__Quickstart__

* `requirements.txt`

* `README.md`

__Data__

`InputArticles` folder : each subfolder corresponds to a year and contains press articles as json files. Articles were published in French newspaper *Ouest France* during the second half of the 20th century. 

* `articles.csv` : articles with meta data in csv format, built from the `InputArticles` folder. Total number of articles is 726.

* `articles_labeled.xlsx` : manual labels for geography and topic.

* `labeled_articles_clean.csv` : with columns containing preprocessed text (article title and body).

* `sample_articles.xlsx` : sample data from 2020 to test the recommandation engine.

__Notebooks__

* `00_BuildDataset.ipynb` : build `articles.csv` from the original json files.

* `01_DataExploration.ipynb`: exploratory analysis of the Ouest France corpus. Contains key statistics, wordcloud visualisations, first NER extractions, LDA.

* `02_GeoParsing.ipynb` : development of a strategy to extract, classify and compare articles based on their GEO entities. Equivalent to the scripts contained in folder `geoparsing`. 

* `02_GeoParsing_applied.ipynb` : application of this strategy.

* `02_TopicModeling.ipynb` : Naïve classifications for geography and topic using specialized or basic tfidf embeddings. K-Means. V1 : lemmatizer.

* `02_TopicModeling_stemmer.ipynb` : V2 : Porter stemmer.

* `03_Recommandation.ipynb` : put in practice the methods developed in the above notebooks. First version of the recommandation pipeline. 


__Geoparsing__

`geoparsing`: scripts corresponding to our attempt to implement an ontological approach to characterize and compare articles based on the geographical entities they contain.

* `geoparsing_datasaver.py`: a script to save various objects containing reference geographical entities (country names, continent-country dictionnary, space vocabulary etc.)

* `geoparsing_utils`: utility functions to extract classify and compare geographical entites from texts, using the reference objects saved before.

* `geoparsing_test`: short demo.


__First pipelines__

`pipeline` folder : scrip equivalent of the notebooks (preprocessing - lemmatizer and stemmer versions, wordcloud, tfidf, lda)

`predictions`folder

* `reco_utils.py`

* `reco_test.py`

__Recommandation scripts__ __TO UPDATE__

* `text_preprocessing.py` : minimal preprocessing needed to run the recommandation pipeline

* `sim_fasttest.py` : script to train and save embeddings and models for the recommandation pipeline (build pipeline v1: tfidf learned on Ouest France articles, reco on Le Monde articles, pretrained fasttext embedding 300d)

* `build_pipeline_1.py`: v2 --> tfidf learnt on Le Monde + Ouest France articles, reco Le Monde + Ouest France articles

* `reco_sim_fasttext.py` : script to actually run the recommandation pipeline (v2)

* `build_pipeline_1bis.py`: v3 --> finetuning of fasttext embedding on Le Monde + Ouest France articles

* `reco_sim_fasttext_finetuned.py` : script to actually run the recommandation pipeline (v3)


__Webhook service__ CONTENT TO ADD

* `webhookservice` folder

* `webhookserviceCSML` folder









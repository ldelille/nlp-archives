# NLP Archives



## Purpose of the project

Starting from **articles from the archives** of a French newspaper, we **build different machine learning models** to compute 
similarities with recent articles. Finding **similarities between recent articles and articles from the archives** is the 
purpose of this repository. Articles recommendations based on semantic similarities and topic recognition.


To enhance our models we recovered some articles from the archives using scraping. 

We develop a chatbot that provides live recommendation from articles. The chatbot parse the article that the user is reading, 
launch similarity computing and propose similar articles from the archives. The associated code has also been move to 
a dedicated repository: [chatbot](https://github.com/ldelille/chatbot-newspaper) 



### Contents of this repository

__Quickstart__

* `requirements.txt`

* `README.md`

__Data__

`InputArticles` folder : each subfolder corresponds to a year and contains press articles as json files. 
Articles were published in French newspaper *Ouest France* during the second half of the 20th century. 


__Notebooks__

We started by exploring our dataset with notebooks. Here is the detail:

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

__Recommandation scripts__ 

* `text_preprocessing.py` : minimal preprocessing needed to run the recommandation pipeline

* `sim_fasttest.py` : script to train and save embeddings and models for the recommandation pipeline (build pipeline v1: tfidf learned on Ouest France articles, reco on Le Monde articles, pretrained fasttext embedding 300d)

* `build_pipeline_1.py`: v2 --> tfidf learnt on Le Monde + Ouest France articles, reco Le Monde + Ouest France articles

* `reco_sim_fasttext.py` : script to actually run the recommandation pipeline (v2)

* `build_pipeline_1bis.py`: v3 --> finetuning of fasttext embedding on Le Monde + Ouest France articles

* `reco_sim_fasttext_finetuned.py` : script to actually run the recommandation pipeline (v3)

* `reco_sim_keyword.py` : variation of `reco_sim_fasttext_finetuned.py` in the case of a search by keyword. Also includes a new search by date functionnality. 











from flask import Flask, request, jsonify, render_template, make_response
import os
import dialogflow
import requests
import json
import pandas as pd

app = Flask(__name__)

from reco_single import RecoArticle


@app.route('/')
def index():
    return render_template('index.html')


def results():
    req = request.get_json(force=True)

    article_number = req.get('queryResult').get('parameters').get('number-integer')
    article_url = req.get('queryResult').get('parameters').get('url')
    print(f"detected article number {article_number} as an input, launching reco...")

    # print(launch_spider(article_url))
    test_article.embed_list = []
    print("test_article", test_article)
    print("result from function is ", test_article.launch_reco_from_id(int(article_number)))
    return {
        'fulfillmentText': 'Nous vous recommandons :' + test_article.launch_reco_from_id(int(article_number))[
            0] + ' à partir de l\'article ' + str(
            article_number)}


# route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))


@app.route('/scraping', methods=['GET', 'POST'])
def launch_scraping():
    req = request.get_json(force=True)
    article_url = req.get('queryResult').get('parameters').get('url')
    print(f"detected article with url {article_url} as an input, launching scraping...")
    test_article.embed_list = []
    params = {
        'spider_name': 'lemonde_single',
        'url': str(article_url)
    }
    response = requests.get('http://localhost:9080/crawl.json', params)
    data = json.loads(response.text)
    test_article.compute_embeddings_from_parsed_article(data['items'][0])
    return {
        'fulfillmentText': 'Nous vous recommandons :' + test_article.launch_reco_from_parsed_article()[
            0] + ' à partir de l\'article ' + str(
            article_url)}


if __name__ == "__main__":
    test_article = RecoArticle()
    test_article.load_models()
    test_article.compute_embeddings_from_sample()
    app.run()

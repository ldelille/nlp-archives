from flask import Flask, request, jsonify, render_template, make_response
import os
import dialogflow
import requests
import json
import pandas as pd

app = Flask(__name__)

from reco_single import RecoArticle
from scraping.crawler_helper import launch_spider

@app.route('/')
def index():
    return render_template('index.html')


def results():
    req = request.get_json(force=True)

    article_number = req.get('queryResult').get('parameters').get('number-integer')
    article_url = req.get('queryResult').get('parameters').get('url')
    print(f"detected article number{article_number} as an input, launching reco...")

    print(launch_spider(article_url))

    return {
        'fulfillmentText': 'Nous vous recommandons :' + test_article.launch_reco_from_id(int(article_number))[
            0] + ' à partir de l\'article ' + str(
            article_number)}


# route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))


if __name__ == "__main__":
    test_article = RecoArticle()
    test_article.load_models()
    app.run()

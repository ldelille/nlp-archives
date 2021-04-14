import flask
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
    print('received a post')


# route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    req = request.get_json(force=True)
    print('res is url', req['is_url']  )
    if req['is_url'] == 'false':
        res = test_article.launch_reco_from_id(int(req['article_id']) - 1)
    else:
        print(f"detected article with url {req['article_id']} as an input, launching scraping...")
        params = {
            'spider_name': 'lemonde_single',
            'url': str(req['article_id'])
        }
        response = requests.get('http://localhost:9080/crawl.json', params)
        data = json.loads(response.text)
        print('data received from scraping', data)
        test_article.compute_embeddings_from_parsed_article(data['items'][0])
        res = test_article.launch_reco_from_parsed_article()
        res["input_article"] = {}
        res["input_article"]["title"] = data['items'][0]["title"]
        res["input_article"]["url"] = data['items'][0]["url"]
        res["input_article"]["date_published"] = data['items'][0]["date_published"]
    resp = make_response(jsonify(res))
    return resp


# route for reco from keywords
@app.route('/keywords', methods=['GET', 'POST'])
def keywords():
    req = request.get_json(force=True)
    req['data'] = ' '.join(req['data']) # unify keywords
    res = test_article.compute_embeddings_from_keywords(req)
    resp = make_response(jsonify(res))
    return resp


if __name__ == "__main__":
    test_article = RecoArticle()
    test_article.load_models()
    test_article.compute_embeddings_from_sample()
    app.run()

# change port: flask run -h localhost -p 3000

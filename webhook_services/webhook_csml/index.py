import flask
from flask import Flask, request, jsonify, render_template, make_response
import os
import dialogflow
import requests
import json
import pandas as pd

app = Flask(__name__)


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
    resp = make_response(jsonify({
        "results": {
            "data": "this is a response from csml flask api",
            "article_id": req['article_id']}
    }))

    return resp


if __name__ == "__main__":
    app.run()

# change port: flask run -h localhost -p 3000

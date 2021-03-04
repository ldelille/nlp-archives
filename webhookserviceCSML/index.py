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
    resp = make_response(jsonify({
        "results": [
            {
                "gender": "male",
                "name": {
                    "title": "Mr",
                    "first": "Sharif",
                    "last": "Ouwens"
                },
                "location": {
                    "street": {
                        "number": 2249,
                        "name": "Groesdijk"
                    },
                    "city": "Steendam",
                    "state": "Friesland",
                    "country": "Netherlands",
                    "postcode": 59339,
                    "coordinates": {
                        "latitude": "46.5676",
                        "longitude": "-166.5982"
                    },
                    "timezone": {
                        "offset": "+9:30",
                        "description": "Adelaide, Darwin"
                    }
                },
                "email": "sharif.ouwens@example.com",
                "login": {
                    "uuid": "d8e3ac28-8637-422b-829b-6e8d4f64b5e0",
                    "username": "organicpeacock453",
                    "password": "brain",
                    "salt": "IzoDikTW",
                    "md5": "81e095fc124ac49413bb3fc8f5c943db",
                    "sha1": "01019c168f866b51bd7c3326e60b5e9d63ff90d2",
                    "sha256": "a24bf43e7b56257f0b0ee68769da8153bc001ce6769982829186f7be061fa8fe"
                },
                "dob": {
                    "date": "1994-09-20T03:34:51.361Z",
                    "age": 27
                },
                "registered": {
                    "date": "2011-02-02T04:53:41.569Z",
                    "age": 10
                },
                "phone": "(293)-263-8027",
                "cell": "(146)-468-4687",
                "id": {
                    "name": "BSN",
                    "value": "59438827"
                },
                "picture": {
                    "large": "https://randomuser.me/api/portraits/men/83.jpg",
                    "medium": "https://randomuser.me/api/portraits/med/men/83.jpg",
                    "thumbnail": "https://randomuser.me/api/portraits/thumb/men/83.jpg"
                },
                "nat": "NL"
            }
        ],
        "info": {
            "seed": "d325d5cb9ed1151b",
            "results": 1,
            "page": 1,
            "version": "1.3"
        }
    }))
    resp.headers['content-type'] = 'application/json; charset=utf-8'
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['vary'] = 'Accept-Encoding'
    return resp


if __name__ == "__main__":
    app.run()

# change port: flask run -h localhost -p 3000

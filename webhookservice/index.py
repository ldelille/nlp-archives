from flask import Flask, request, jsonify, render_template, make_response
import os
import dialogflow
import requests
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

# function for responses
def results():
    # build a request object
    req = request.get_json(force=True)

    # fetch action from json
    action = req.get('queryResult').get('action')

    # return a fulfillment response
    return {'fulfillmentText': 'This is a response from webhook.'}

# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))

if __name__ == "__main__":
    app.run()

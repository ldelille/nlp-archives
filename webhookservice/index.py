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
    req = request.get_json(force=True)

    article_number = req.get('queryResult').get('parameters').get('number-integer')
    print(article_number)
    labeled_df = pd.read_csv("../labeled_articles_clean.csv")
    print(    labeled_df.clean_title[0])
    return {'fulfillmentText': 'Nous vous recommandons :'+ labeled_df.clean_title[0]+ ' à partir de l\'article '+ str(article_number)}

# route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))

if __name__ == "__main__":
    app.run()

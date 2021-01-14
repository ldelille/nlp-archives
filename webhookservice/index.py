from flask import Flask, request, jsonify, render_template
import os
import dialogflow
import requests
import json

app = Flask(__name__)


@app.route('/')
def index():
    return


if __name__ == "__main__":
    app.run()

import spacy
from spacy.matcher import PhraseMatcher
import json
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from spacymoji import Emoji

nlp = spacy.load('en_core_web_lg')

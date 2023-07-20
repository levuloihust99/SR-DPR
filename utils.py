import os 
import json
import logging
from typing import Text
from dpr.utils.logging_utils import add_color_formater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)

def load_corpus_dataset(corpus_path: Text):
    if os.path.splitext(corpus_path)[1]== "json":
        with open(corpus_path, "r") as reader:
            corpus = json.load(reader)
    else:
        corpus = []
        with open(corpus_path, "r") as reader:
            for line in reader:
                corpus.append(json.loads(line.strip()))
    return corpus
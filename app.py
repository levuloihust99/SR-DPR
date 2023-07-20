import json
import requests
import argparse
import logging
from transformers import BertTokenizer
import torch 

from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json as sanic_json
from datetime import datetime

from libs.utils.utils import add_color_formater
# from libs.utils.setup import setup_memory_growth
from libs.faiss_indexer import DenseFlatIndexer
from libs.utils.utils import load_query_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)

app = Sanic(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.middleware("request")
def before_request_func(request):
    request.ctx.start_time = datetime.now()


@app.middleware("response")
def after_request_func(request, response):
    logger.info(f"Total processing time: {datetime.now() - request.ctx.start_time}")
    return response


@app.route("/retrieve", methods=["POST"])
def retrieve(request):
    data = request.json
    query = data['query']
    logger.debug("\x1b[38;5;11;1mReceived query: {}\x1b[0m".format(query))
    if app.ctx.args.do_lowercase:
        query = query.lower()
    if app.ctx.args.do_segment:
        headers = {'Content-Type': 'application/json'}
        segment_resp = requests.post(app.ctx.args.segment_endpoint, data=json.dumps({'sentence': query}), headers=headers)
        query = segment_resp.json()['sentence']

    top_docs = data.get('top_docs', 10)
    query_tokens = app.ctx.tokenizer.tokenize(query)
    logger.debug("\x1b[38;5;11;1mTokenized query: {}\x1b[0m".format(query_tokens))
    query_tokens = [app.ctx.tokenizer.cls_token] + query_tokens + [app.ctx.tokenizer.sep_token]
    query_ids = app.ctx.tokenizer.convert_tokens_to_ids(query_tokens)
    outputs = app.ctx.query_encoder(input_ids=torch.tensor([query_ids]), return_dict=True)
    query_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

    retrieval_results = app.ctx.indexer.search_knn(query_embedding, top_docs=top_docs)
    metas = retrieval_results[0][0]
    scores = retrieval_results[0][1]
    scores = [float(score) for score in scores]
    retrieval_results = [{**meta, "score": score} for meta, score in zip(metas, scores)]

    return sanic_json(retrieval_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["bert", "roberta"], default="bert")
    parser.add_argument("--tokenizer-path", default="NlpHUST/vibert4news-base-cased")
    parser.add_argument("--tokenizer-config", default=r"{}")
    parser.add_argument("--pretrained-model-path", default="NlpHUST/vibert4news-base-cased")
    parser.add_argument("--checkpoint-path", default="tf/dpr_biencoder.1000-1")
    parser.add_argument("--torch-checkpoint-path", default="checkpoint/dpr_biencoder.1000")
    parser.add_argument("--index-path", default="indexes")
    parser.add_argument("--do-segment", action='store_true', default=False)
    parser.add_argument("--do-lowercase", action="store_true", default=True)
    parser.add_argument("--segment-endpoint", default="http://localhost:8088/segment")
    parser.add_argument("--port", type=int, default=5666)
    args = parser.parse_args()
    app.ctx.args = args

    # setup_memory_growth()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    app.ctx.tokenizer = tokenizer

    indexer = DenseFlatIndexer()
    indexer.deserialize(args.index_path)
    app.ctx.indexer = indexer
    
    query_encoder = load_query_encoder(checkpoint_path = args.torch_checkpoint_path, pretrained_model_path=args.pretrained_model_path)
    app.ctx.query_encoder = query_encoder


if __name__ == "__main__":
    main()
    HOST = "0.0.0.0"
    PORT = app.ctx.args.port
    DEBUG = False
    THREADED = False
    app.run(host=HOST, port=PORT, debug=DEBUG, workers=1)

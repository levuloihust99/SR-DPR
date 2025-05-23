#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
import jsonlines
import re
import requests
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T, tensor
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer, DistributedFaissDenseIndexer
from dpr.models.hf_models import get_bert_tensorizer
from dpr.utils.logging_utils import add_color_formater

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)
add_color_formater(logging.root)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def parse_qa_jsonl_file(location) -> Iterator[Tuple[str, List[str]]]:
    with jsonlines.open(location) as reader:
        for row in reader:
            question = row['question']
            answers = row['answers']
            yield question, answers


def validate(passages: Dict[object, Tuple[str, str]], corpus_endpoint: str, answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str) -> List[List[bool]]:
    match_stats = calculate_matches(passages, corpus_endpoint, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, 'rt') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t', )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != 'id':
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results(passages: Dict[object, Tuple[str, str]], corpus_endpoint: str, questions: List[str], answers: List[List[str]],
                 top_passages_and_scores: List[Tuple[List[object], List[float]]], per_question_hits: List[List[bool]],
                 out_file: str
                 ):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        doc_ids = results_and_scores[0]
        if passages is not None:
            local_passages = {doc_id: passages[doc_id] for doc_id in doc_ids}
        else:
            resp = requests.post(corpus_endpoint, data=json.dumps({"doc_ids": doc_ids}), headers={"Content-Type": "application/json"})
            local_passages = resp.json()
            local_passages = {doc_id: (doc["text"], doc["title"]) for doc_id, doc in zip(doc_ids, local_passages)}
        hits = per_question_hits[i]
        docs = [local_passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    do_question_embedding = not (args.q_encoding_path and not args.re_encode_q and os.path.exists(args.q_encoding_path))
    if do_question_embedding:
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

        encoder = encoder.question_model

        encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                                args.local_rank,
                                                args.fp16)
        encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info('Loading saved model state ...')

        prefix_len = len('question_model.')
        question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                                key.startswith('question_model.')}
        model_to_load.load_state_dict(question_encoder_state)
        vector_size = model_to_load.get_out_size()
        logger.info('Encoder vector_size=%d', vector_size)
    else:
        tensorizer = None
        encoder = None

    # get questions & answers
    questions = []
    question_answers = []

    parse_fn = parse_qa_csv_file if args.qa_format == 'csv' else parse_qa_jsonl_file
    for ds_item in parse_fn(args.qa_file):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    if not do_question_embedding:
        questions_tensor = torch.load(args.q_encoding_path)
    else:
        questions_tensor = retriever.generate_question_vectors(questions)
        if args.encode_q_and_save:
            torch.save(questions_tensor, args.q_encoding_path)

    # finished encoding, exit
    if args.encode_q_and_save:
        return

    vector_size = questions_tensor.shape[-1]
    index_paths = args.index_paths.split(',')
    index = DistributedFaissDenseIndexer(
        vector_sz=vector_size,
        indexers=None,
        index_paths=index_paths,
        buffer_size=args.index_buffer
    )
    # if args.hnsw_index:
    #     index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    # else:
    #     index = DenseFlatIndexer(vector_size, args.index_buffer)
    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # index all passages
    # input_paths = os.listdir(args.encoded_ctx_dir)
    # input_paths = sorted(input_paths, key=lambda x: int(re.search(r"wikipedia_passages_(\d+)\.pkl", x).group(1)))
    # input_paths = [os.path.join(args.encoded_ctx_dir, f) for f in input_paths]

    # index_path = "_".join(input_paths[0].split("_")[:-1])
    # index_path = args.index_path
    # logger.info("Index path: '{}'".format(index_path))
    # if args.save_or_load_index and (os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")):
    #     retriever.index.deserialize_from(index_path)
    # else:
    #     logger.info('Reading all passages data from files: %s', input_paths)
    #     retriever.index.index_data(input_paths)
    #     if args.save_or_load_index:
    #         retriever.index.serialize(index_path)

    # get top k results
    logger.info("Index searching...")
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

    all_passages = None
    corpus_endpoint = None
    if not args.remote_corpus:
        all_passages = load_passages(args.ctx_file)
    else:
        corpus_endpoint = args.corpus_endpoint

    # if len(all_passages) == 0:
    #     raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    validation_results = validate(all_passages, corpus_endpoint, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match)
    questions_doc_hits = validation_results.questions_doc_hits
    top_hits = validation_results.top_k_hits
    top_hits = [hit / len(question_answers) for hit in top_hits]
    pretty_results = ["Top {:03d}: {}".format(idx + 1, hit) for idx, hit in enumerate(top_hits)]
    pretty_results = "\n".join(pretty_results)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    with open(os.path.join(args.result_dir, "top_hits.txt"), "w") as writer:
        writer.write(pretty_results + "\n")
    with open(os.path.join(args.result_dir, "match_matrix.txt"), "w") as writer:
        writer.write(json.dumps(questions_doc_hits) + "\n")

    if args.out_file:
        logger.info("Saving retrieval results...")
        save_results(all_passages, corpus_endpoint, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--qa_format', default='csv', choices=['csv', 'jsonl'],
                        help="Format of the file containing test questions and answers")
    parser.add_argument('--ctx_file', default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    # parser.add_argument('--encoded_ctx_dir', type=str, default=None,
    #                     help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .json file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=200, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

    parser.add_argument("--encode_q_and_save", action='store_true')
    parser.add_argument("--re_encode_q", action='store_true')
    parser.add_argument("--q_encoding_path")
    parser.add_argument("--remote_corpus", action='store_true', help="If enabled, do not load corpus into RAM but make request to corpus endpoint")
    parser.add_argument("--corpus_endpoint", help="Corpus endpoint to get documents")
    parser.add_argument("--index_paths", help="Paths to save or load the FAISS indexes, comma seperated")
    parser.add_argument("--result_dir", default="results", help="Path to save top_k_hits and questions_doc_hits result")

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)

    if args.encode_q_and_save and args.q_encoding_path is None:
        raise ValueError(f'Requires q_encoding_path when encoding question')

    main(args)

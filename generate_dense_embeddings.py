#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib

import argparse
import csv
import logging
import pickle
from typing import Iterable, Iterator, List, Tuple
from tqdm import tqdm
from itertools import tee

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint,move_to_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class CtxDataset(Dataset):
    def __init__(self, ctx_rows: List[Tuple[object, str, str]], tensorizer: Tensorizer, insert_title: bool = True):
        self.rows = ctx_rows
        self.tensorizer = tensorizer
        self.insert_title = insert_title

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        ctx = self.rows[item]

        return self.tensorizer.text_to_tensor(ctx[1], title=ctx[2] if self.insert_title else None)


class IterableCtxDataset(IterableDataset):
    def __init__(self, ctx_rows: Iterator[Tuple[object, str, str]]):
        self.ctx_rows = ctx_rows
    
    def __iter__(self):
        return self.ctx_rows


def no_op_collate(xx: List[object]):
    return xx


def get_tensorize_collate_fn(tensorizer, insert_title: bool = True):
    def tensorize_collate_fn(items: List[object]):
        return [tensorizer.text_to_tensor(ctx[1], title=ctx[2] if insert_title else None) for ctx in items]
    return tensorize_collate_fn


def gen_ctx_vectors(ctx_rows: List[Tuple[object, str, str]], model: nn.Module, tensorizer: Tensorizer,
                    insert_title: bool = True, fp16: bool = False, num_vectors_per_file: int = 100000, shard_size: int = None) -> List[Tuple[object, np.array]]:
    bsz = args.batch_size
    total = 0
    results = []
    backup_ctx_rows = None
    iter_state = 0

    corpus_size = None
    if isinstance(ctx_rows, list):
        shard_size = len(ctx_rows)
        dataset = CtxDataset(ctx_rows, tensorizer, insert_title)
        loader = DataLoader(
            dataset, shuffle=False, num_workers=2, collate_fn=no_op_collate, drop_last=False, batch_size=bsz)
    else:
        ctx_rows, backup_ctx_rows = tee(ctx_rows)
        dataset = IterableCtxDataset(ctx_rows)
        collate_fn = get_tensorize_collate_fn(tensorizer, insert_title=insert_title)
        loader = DataLoader(
            dataset, shuffle=False, collate_fn=collate_fn, drop_last=False, batch_size=bsz)

    for batch_id, batch_token_tensors in tqdm(enumerate(loader), total=shard_size):
        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0),args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch),args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch),args.device)
        with torch.no_grad():
            if fp16:
                with autocast():
                    _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
            else:
                _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.float().cpu()

        batch_start = batch_id*bsz
        if backup_ctx_rows is None:
            ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]
        else:
            ctx_ids = []
            while iter_state < batch_start + bsz:
                r = next(backup_ctx_rows)
                ctx_ids.append(r[0])
                iter_state += 1

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy())
            for i in range(out.size(0))
        ])

        if len(results) >= num_vectors_per_file:
            yield results[:num_vectors_per_file]
            results = results[num_vectors_per_file:]

    return results


def get_corpus(
    corpus_path: str,
    start_idx: int,
    end_idx: int,
    corpus_reader: str = 'bulk'
):
    logger.info('Producing encodings for passages range: %d to %d', start_idx, end_idx)
    with open(corpus_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        if corpus_reader == 'bulk':
            rows = []
            rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])
            return rows[start_idx: end_idx]
        elif corpus_reader == 'sequential':
            if end_idx < 0:
                end_idx = float('inf')
            for idx, row in enumerate(reader):
                if start_idx <= idx - 1 < end_idx:
                    yield row[0], row[1], row[2]


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)
    
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    prefix_len = len('ctx_model.')
    ctx_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                 key.startswith('ctx_model.')}
    model_to_load.load_state_dict(ctx_state)

    logger.info('reading data from file=%s', args.ctx_file)

    rows = get_corpus(
        corpus_path=args.ctx_file,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        corpus_reader=args.ctx_file_reader
    )

    data_generator = gen_ctx_vectors(
        rows, encoder, tensorizer, True, fp16=args.fp16, num_vectors_per_file=args.num_vectors_per_file, shard_size=args.ctx_corpus_size)
    for i, data in enumerate(data_generator):
        file = args.out_file + '_' + str(i) + '.pkl'
        pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        logger.info('Writing results to %s' % file)
        with open(file, mode='wb') as f:
            pickle.dump(data, f)
        logger.info('Total passages processed %d. Written to %s', len(data), file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--ctx_file_reader', type=str, choices=['bulk', 'sequential'], default='bulk')
    parser.add_argument('--ctx_corpus_size', type=int, default=None)
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output file path to write results to')
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of data")
    parser.add_argument('--end_idx', type=int, default=-1, help="End index of data")
    parser.add_argument('--num_vectors_per_file', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)

    
    main(args)

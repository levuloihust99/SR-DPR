#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import glob
import logging
import os
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

logger = logging.getLogger()

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])


def setup_for_distributed_mode(model: nn.Module, optimizer: torch.optim.Optimizer, device: object, n_gpu: int = 1,
                               local_rank: int = -1,
                               fp16: bool = False,
                               fp16_opt_level: str = "O1") -> (nn.Module, torch.optim.Optimizer):
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    return model, optimizer


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def move_to_device(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_device(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_device(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_device(sample, device)


def get_schedule_linear(optimizer, warmup_steps, training_steps, steps_shift=0, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def init_weights(modules: List):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def get_model_file(args, file_prefix) -> str:
    if args.model_file and os.path.exists(args.model_file):
        return args.model_file

    out_cp_files = glob.glob(os.path.join(args.output_dir, file_prefix + '*')) if args.output_dir else []
    logger.info('Checkpoint files %s', out_cp_files)
    model_file = None

    if len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
    return model_file


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    # restore RNG
    cpu_state = state_dict.pop('cpu_state', None)
    cuda_states = state_dict.pop('cuda_states', None)
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        if torch.cuda.is_available() > 0:
            device_id = torch.cuda.current_device()
            if device_id < len(cuda_states):
                try:
                    torch.cuda.set_rng_state(cuda_states[device_id])
                except Exception:
                    logger.info("Invalid RNG state restored from checkpoint file '{}'".format(model_file))
    return CheckpointState(**state_dict)

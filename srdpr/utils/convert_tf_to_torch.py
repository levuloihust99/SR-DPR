"""This script converts a TF checkpoint to PyTorch checkpoint (for SR-DPR)."""

import re
import time
import torch
import logging

import argparse
import numpy as np
import tensorflow as tf
from transformers import BertModel, TFBertModel

from srdpr.utils.logging_utils import add_color_formatter

logger = logging.getLogger(__name__)


def convert_bert_from_tf_to_torch(tf_bert: TFBertModel, torch_bert: BertModel):
    tf_weights = tf_bert.weights
    tf_weights = {w.name: w.numpy() for w in tf_weights}
    torch_state_dict = {"embeddings.position_ids": torch_bert.embeddings.position_ids}
    for tf_name, w in tf_weights.items():
        torch_name, transpose = get_torch_name_from_tf_name(tf_name)
        torch_state_dict[torch_name] = torch.tensor(w) if transpose is False else torch.tensor(w.T)
    torch_bert.load_state_dict(torch_state_dict)


def get_torch_name_from_tf_name(tf_name):
    tf_name = re.match(r"^tf_bert_model.*?/bert/(?P<name_match>.*):0$", tf_name).group("name_match")
    if tf_name.startswith("embeddings/"):
        if tf_name == "embeddings/word_embeddings/weight":
            torch_name = "embeddings.word_embeddings.weight"
        elif tf_name == "embeddings/position_embeddings/embeddings":
            torch_name = "embeddings.position_embeddings.weight"
        elif tf_name == "embeddings/token_type_embeddings/embeddings":
            torch_name = "embeddings.token_type_embeddings.weight"
        elif tf_name == "embeddings/LayerNorm/gamma":
            torch_name = "embeddings.LayerNorm.weight"
        elif tf_name == "embeddings/LayerNorm/beta":
            torch_name = "embeddings.LayerNorm.bias"
        else:
            raise Exception("Unexpected weight: {}".format(tf_name))
        return torch_name, False
    elif tf_name.startswith("encoder"):
        match = re.match(r"^encoder/layer_\._(?P<layer>\d+)/(?P<sublayer>.*?)/(?P<remain>.*)$", tf_name)
        layer = match.group("layer")
        sublayer = match.group("sublayer")
        remain = match.group("remain")
        if sublayer == "attention":
            if remain.startswith("self"):
                match = re.match(r"^self/(?P<qkv>.*?)/(?P<wandb>.*)$", remain)
                qkv = match.group("qkv")
                wandb = match.group("wandb")
                torch_wandb = "weight" if wandb == "kernel" else "bias"
                transpose = True if wandb == "kernel" else False
                return "encoder.layer.{}.attention.self.{}.{}".format(layer, qkv, torch_wandb), transpose
            elif remain.startswith("output"):
                if remain == "output/dense/kernel":
                    return "encoder.layer.{}.attention.output.dense.weight".format(layer), True
                elif remain == "output/dense/bias":
                    return "encoder.layer.{}.attention.output.dense.bias".format(layer), False
                elif remain == "output/LayerNorm/gamma":
                    return "encoder.layer.{}.attention.output.LayerNorm.weight".format(layer), False
                elif remain ==  "output/LayerNorm/beta":
                    return "encoder.layer.{}.attention.output.LayerNorm.bias".format(layer), False
                else:
                    raise Exception("Unexpected weight: {}".format(tf_name))
            else:
                raise Exception("Unexpected weight: {}".format(tf_name))
        elif sublayer == "intermediate":
            if remain == "dense/kernel":
                return "encoder.layer.{}.intermediate.dense.weight".format(layer), True
            elif remain == "dense/bias":
                return "encoder.layer.{}.intermediate.dense.bias".format(layer), False
            else:
                raise Exception("Unexpected weight: {}".format(tf_name))
        elif sublayer == "output":
            if remain == "dense/kernel":
                return "encoder.layer.{}.output.dense.weight".format(layer), True
            elif remain == "dense/bias":
                return "encoder.layer.{}.output.dense.bias".format(layer), False
            elif remain == "LayerNorm/gamma":
                return "encoder.layer.{}.output.LayerNorm.weight".format(layer), False
            elif remain == "LayerNorm/beta":
                return "encoder.layer.{}.output.LayerNorm.bias".format(layer), False
    elif tf_name.startswith("pooler"):
        if tf_name == "pooler/dense/kernel":
            return "pooler.dense.weight", True
        elif tf_name == "pooler/dense/bias":
            return "pooler.dense.bias", False
        else:
            raise Exception("Unexpected weight: {}".format(tf_name))
    else:
        raise Exception("Unexpected weight: {}".format(tf_name))


def validate_equivalence(tf_bert: TFBertModel, torch_bert: BertModel):
    tf_input_ids = tf.constant([[1, 2, 3]])
    tf_outputs = tf_bert(input_ids=tf_input_ids, training=False, return_dict=True)
    tf_sequence_output = tf_outputs.last_hidden_state
    tf_pooled_output = tf_sequence_output[:, 0, :]

    torch_input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch_outputs = torch_bert(input_ids=torch_input_ids, return_dict=True)
        torch_sequence_output = torch_outputs.last_hidden_state
        torch_pooled_output = torch_sequence_output[:, 0, :]
    
    tf_pooled_output = tf_pooled_output.numpy()
    torch_pooled_output = torch_pooled_output.numpy()

    is_equivalence = np.all(np.abs(tf_pooled_output - torch_pooled_output) < 1e-4)
    return bool(is_equivalence)


def run(args):
    logger.info("Loading TF pretrained...")
    t0 = time.perf_counter()
    question_model = TFBertModel.from_pretrained(args.pretrained_model_name)
    ctx_model = TFBertModel.from_pretrained(args.pretrained_model_name)
    logger.info("Done loading TF pretrained in {}s".format(time.perf_counter() - t0))

    tf_model = tf.train.Checkpoint(query_encoder=question_model, context_encoder=ctx_model)
    ckpt = tf.train.Checkpoint(model=tf_model)

    logger.info("Restoring checkpoint from {} ...".format(args.tf_ckpt_path))
    t0 = time.perf_counter()
    ckpt.restore(args.tf_ckpt_path).expect_partial()
    logger.info("Done restoring checkpoint in {}s".format(time.perf_counter() - t0))

    logger.info("Loading PyTorch pretrained...")
    t0 = time.perf_counter()
    torch_question_model = BertModel.from_pretrained(args.pretrained_model_name)
    torch_ctx_model = BertModel.from_pretrained(args.pretrained_model_name)
    logger.info("Done loading PyTorch checkpoint in {}s".format(time.perf_counter() -  t0))

    convert_bert_from_tf_to_torch(question_model, torch_question_model)
    convert_bert_from_tf_to_torch(ctx_model, torch_ctx_model)

    logger.info("Validating Tensorflow - PyTorch equivalence...")
    t0 = time.perf_counter()
    assert validate_equivalence(question_model, torch_question_model)
    assert validate_equivalence(ctx_model, torch_ctx_model)
    logger.info("Done validating Tensorflow - PyTorch equivalence in {}s".format(time.perf_counter() - t0))

    logger.info("Saving PyTorch checkpoint into {} ...".format(args.torch_ckpt_path))
    torch_state_dict = {}
    for k, v in torch_question_model.state_dict().items():
        torch_state_dict["question_model.{}".format(k)] = v
    for k, v in torch_ctx_model.state_dict().items():
        torch_state_dict["ctx_model.{}".format(k)] = v
    torch.save({"model_dict": torch_state_dict}, args.torch_ckpt_path)
    logger.info("Done saving PyTorch checkpoint in {}s".format(time.perf_counter() - t0))


def main():
    logging.basicConfig(level=logging.INFO)
    add_color_formatter(logging.root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", default="bert-base-uncased")
    parser.add_argument("--tf_ckpt_path", required=True)
    parser.add_argument("--torch_ckpt_path", required=True)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()

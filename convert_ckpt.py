import os
import argparse
import torch
import tensorflow as tf
from transformers import BertModel, TFBertModel


def separate(args):
    # create output dir if necessary
    question_model_output_dir = os.path.join(args.separate_dir, "question_model")
    ctx_model_output_dir = os.path.join(args.separate_dir, "ctx_model")
    if not os.path.exists(question_model_output_dir):
        os.makedirs(question_model_output_dir)
    if not os.path.exists(ctx_model_output_dir):
        os.makedirs(ctx_model_output_dir)

    saved_state = torch.load(args.torch_cp, map_location=lambda s, t: s)
    saved_state = saved_state['model_dict']
    question_prefix = "question_model."
    ctx_prefix = "ctx_model."
    question_model_state = {
        k[len(question_prefix):]: v
        for k, v in saved_state.items()
        if k.startswith(question_prefix)
    }
    # assert len(question_model_state) == 199
    question_model = BertModel.from_pretrained(args.pretrained_model_path)
    question_model.load_state_dict(question_model_state, strict= False)
    question_model.save_pretrained(question_model_output_dir)

    ctx_model_state = {
        k[len(ctx_prefix):]: v
        for k, v in saved_state.items()
        if k.startswith(ctx_prefix)
    }
    # assert len(ctx_model_state) == 199
    ctx_model = BertModel.from_pretrained(args.pretrained_model_path)
    ctx_model.load_state_dict(ctx_model_state,  strict= False)
    ctx_model.save_pretrained(ctx_model_output_dir)
  

def convert(args):
    question_model_path = os.path.join(args.separate_dir, "question_model")
    ctx_model_path = os.path.join(args.separate_dir, "ctx_model")
    question_model = TFBertModel.from_pretrained(question_model_path, from_pt=True)
    ctx_model = TFBertModel.from_pretrained(ctx_model_path, from_pt=True)
    model = tf.train.Checkpoint(query_encoder=question_model, context_encoder=ctx_model)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.save(args.tf_cp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-cp",
                        help="Path to PyTorch checkpoint")
    parser.add_argument("--tf-cp",
                        help="Path to TensorFlow checkpoint, needs to be a file prefix.")
    parser.add_argument("--separate-dir",
                        help="Path to save separated question model and ctx model.")
    parser.add_argument("--mode", choices=['separate', 'convert'],
                        help="If mode=separate, separate question_model and ctx_model from the PyTorch checkpoint. "
                            "If mode=convert, convert the separated saved model to TensorFlow checkpoint.")
    parser.add_argument("--pretrained-model-path", default="bert-base-uncased")
    args = parser.parse_args()
    if args.mode == 'separate':
        separate(args)
    elif args.mode == 'convert':
        convert(args)
    else:
        separate(args)
        convert(args)
        


if __name__ == "__main__":
    main()

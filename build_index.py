import os
import json
import torch
import argparse

from tqdm import tqdm
from typing import Text, Dict
from transformers import BertModel, BertTokenizer
from libs.faiss_indexer import DenseFlatIndexer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--pth-checkpoint", required=False, default='checkpoint/dpr_biencoder.1000', 
                        help = 'Path to pytorch checkpoint')
    parser.add_argument("--use-cuda", default=False, action="store_true")
    args = parser.parse_args()
    # Load Config
    with open(args.config_file) as reader:
        cfg = json.load(reader)
    cfg = argparse.Namespace(**cfg)
    # Load Context Model
    saved_state = torch.load(args.pth_checkpoint, map_location=lambda s, t: s)
    saved_state = saved_state['model_dict']
    ctx_prefix = "ctx_model."

    # Context Model
    ctx_model_state = {
        k[len(ctx_prefix):]: v
        for k, v in saved_state.items()
        if k.startswith(ctx_prefix)
    }
    # assert len(ctx_model_state) == 199
    ctx_model = BertModel.from_pretrained(cfg.pretrained_model_path)
    ctx_model.load_state_dict(ctx_model_state,  strict=False)
    ctx_model.eval()
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        ctx_model.to('cuda')
    else:
        device ='cpu'
        
    tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_model_path)

    corpus_meta = load_corpus_dataset(cfg.corpus_meta_path)
    corpus_indexed = load_corpus_dataset(cfg.corpus_index_path)
    data_to_be_indexed = []
    with torch.no_grad():
        for context, ctx_meta in tqdm(zip(corpus_indexed, corpus_meta)):
            all_emb = []
            if context["title"]:
                inputs = tokenizer(context["title"], text_pair=context["text"], return_tensors="pt", truncation=True, max_length=512)
            else:
                inputs = tokenizer(context["text"], return_tensors="pt", truncation=True, max_length=512) 
            outputs = ctx_model(input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device), return_dict=True)
            sequence_output = outputs.last_hidden_state # [bsz, seq_len, hidden_size] -> [1, 20, 768]
            pooled_output = sequence_output[:, 0, :]
            all_emb.append(pooled_output)
            all_emb = torch.cat(all_emb, dim=0)
            data_to_be_indexed.append((ctx_meta,  all_emb[0].cpu().numpy()))

    indexer = DenseFlatIndexer()
    indexer.init_index(vector_sz=768) #(vector_sz=embeddings[0].shape[0]
    indexer.index_data(data_to_be_indexed)
    indexer.serialize(cfg.index_path)


if __name__ == "__main__":
    main()
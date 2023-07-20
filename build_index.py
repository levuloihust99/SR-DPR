import argparse
import json
from typing import Text
from dpr.indexer.faiss_indexers import DenseFlatIndexer
from utils import load_corpus_dataset, logger
import torch 
from transformers import BertTokenizer,BertModel
from tqdm import tqdm

def build_index(args, **kwargs):
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
    assert len(ctx_model_state) == 199
    ctx_model = BertModel.from_pretrained(cfg.pretrained_model_path)
    ctx_model.load_state_dict(ctx_model_state,  strict=False)
    ctx_model.eval()
    
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
            outputs = ctx_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, return_dict=True)
            sequence_output = outputs.last_hidden_state # [bsz, seq_len, hidden_size] -> [1, 20, 768]
            pooled_output = sequence_output[:, 0, :]
            all_emb.append(pooled_output)
            all_emb = torch.cat(all_emb, dim=0)
            data_to_be_indexed.append((ctx_meta,  all_emb[0]))

    indexer = DenseFlatIndexer()
    indexer.init_index(vector_sz=768) #(vector_sz=embeddings[0].shape[0]
    indexer.index_data(data_to_be_indexed)
    indexer.serialize(cfg.index_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--pth-checkpoint", required=False, default='checkpoint/dpr_biencoder.1000', 
                        help = 'Path to pytorch checkpoint')
    args = parser.parse_args()
    build_index(args)

if __name__ == "__main__":
    main()

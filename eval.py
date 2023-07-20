import torch 
import logging
import argparse
from tqdm import tqdm 
import os 
from utils import load_corpus_dataset, logger
from transformers import BertTokenizer, BertModel
from libs.faiss_indexer import DenseFlatIndexer

def evaluate(model_path: str, corpus_index_path, pretrained_model_path):
    """Evaluate DPR Model:
    - Load Context Model, Question Model 
    - Build Index 
    - Query and Eval by Top@k
    """
    # Load Model
    ## Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    ## Load Context Model
    saved_state = torch.load(model_path, map_location=lambda s, t: s)
    saved_state = saved_state['model_dict']
    ctx_prefix = "ctx_model."
    
    ctx_model_state = {
        k[len(ctx_prefix):]: v
        for k, v in saved_state.items()
        if k.startswith(ctx_prefix)
    }
    assert len(ctx_model_state) == 199
    ctx_model = BertModel.from_pretrained(pretrained_model_path)
    ctx_model.load_state_dict(ctx_model_state,  strict=False)
    ctx_model.eval()
    
    ##Load Question Model 
    # saved_state = torch.load(model_path, map_location=lambda s, t: s)
    # saved_state = saved_state['model_dict']
    question_prefix = "question_model."
    question_model_state = {
            k[len(question_prefix):]: v
            for k, v in saved_state.items()
            if k.startswith(question_prefix)
        }
    assert len(question_model_state) == 199
    question_model = BertModel.from_pretrained(pretrained_model_path)
    question_model.load_state_dict(question_model_state, strict= False)
    
    #Build Indexes
    ##Load data
    '''Corpus Dataset'''
    corpus_indexed = load_corpus_dataset(corpus_index_path)
    
    data_to_be_indexed = []
    with torch.no_grad():
        for context, ctx_meta in tqdm(corpus_indexed):
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
    index_path = os.path.dirname(corpus_index_path)
    indexer.serialize(index_path)
    TP=FP=0
    top_docs = 10
    for data in corpus_indexed:
        query = data['question']
        query_tokens = tokenizer.tokenize(query)
        logger.debug("\x1b[38;5;11;1mTokenized query: {}\x1b[0m".format(query_tokens))
        query_tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token]
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        outputs = question_model(input_ids=torch.tensor([query_ids]), return_dict=True)
        query_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

        retrieval_results = indexer.search_knn(query_embedding, top_docs=top_docs)
        metas = retrieval_results[0][0]
        # scores = retrieval_results[0][1]
        # scores = [float(score) for score in scores]
        # retrieval_results = [{**meta, "score": score} for meta, score in zip(metas, scores)]   
        true_id = data['article_id']
        retrieval_id = [x['article_id'] for x in metas]
        if true_id in retrieval_id:
            TP +=1
        else:
            FP +=1
    print(f"Accuracy Top@{top_docs}: {TP/(TP+FP)}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth-checkpoint", required=False, default='checkpoint/dpr_biencoder.1000', 
                        help = 'Path to pytorch checkpoint')
    parser.add_argument("--pretrained-model-path", required=False, default="NlpHUST/vibert4news-base-cased",
                        help = 'Path to pytorch checkpoint')
    parser.add_argument("--corpus-index-path", required=True, help="path to copus data, jsonl")
    args = parser.parse_args()
    
    evaluate(args.pth_checkpoint, args.corpus_index_path, args.pretrained_model_path)
    
if __name__=='__main__':
    main()
import os
import pickle
import argparse
import multiprocessing as mp

from tqdm import tqdm
from transformers import BertTokenizer

from srdpr.utils.helpers import recursive_apply
from srdpr.data_helpers.datasets import ByteDataset
from dpr.utils.data_utils import normalize_question

TOKENIZER_MAPPINGS = {
    "bert": BertTokenizer,
}


def tokenize_item(item):
    if not hasattr(tokenize_item, 'tokenizer'):
        raise Exception("You must set tokenizer for this function by setting its attribute 'tokenizer'.")
    tokenizer = getattr(tokenize_item, 'tokenizer')
    questions = item['questions']
    tokenized_questions = []
    for q in questions:
        q = normalize_question(q).strip()
        q_tokens = tokenizer.tokenize(q)
        q_ids = tokenizer.convert_tokens_to_ids(q_tokens)
        tokenized_questions.append(q_ids)

    positive_contexts = item['positive_contexts']
    tokenized_positive_contexts = []
    for p in positive_contexts:
        recursive_apply(p, fn=lambda x: x.strip())
        p_title = p['title']
        p_title_tokens = tokenizer.tokenize(p_title)
        p_title_ids = tokenizer.convert_tokens_to_ids(p_title_tokens)
        p_text = p['text']
        p_text_tokens = tokenizer.tokenize(p_text)
        p_text_ids = tokenizer.convert_tokens_to_ids(p_text_tokens)
        tokenized_positive_contexts.append({'title': p_title_ids, 'text': p_text_ids})

    hardneg_contexts = item['hardneg_contexts']
    tokenized_hardneg_contexts = []
    for p in hardneg_contexts:
        recursive_apply(p, fn=lambda x: x.strip())
        p_title = p['title']
        p_title_tokens = tokenizer.tokenize(p_title)
        p_title_ids = tokenizer.convert_tokens_to_ids(p_title_tokens)
        p_text = p['text']
        p_text_tokens = tokenizer.tokenize(p_text)
        p_text_ids = tokenizer.convert_tokens_to_ids(p_text_tokens)
        tokenized_hardneg_contexts.append({'title': p_title_ids, 'text': p_text_ids})

    out_item = {
        'sample_id': item['sample_id'],
        'questions': tokenized_questions,
        'positive_contexts': tokenized_positive_contexts,
        'hardneg_contexts': tokenized_hardneg_contexts
    }
    return out_item


def worker_func(feeding_queue, fetching_queue):
    while True:
        item = feeding_queue.get()
        out_item = tokenize_item(item)
        fetching_queue.put(out_item)


def feeding(feeding_queues, args):
    dataset = ByteDataset(data_path=args.bytedataset_path, idx_record_size=6)
    start, stop = args.portion.split(":")
    start = int(start)
    stop = int(stop)
    if stop == -1:
        stop = len(dataset)
    stop = min(len(dataset), stop)
    iterator = get_dataset_iterator(dataset, start, stop)
    
    cyclic_idx = 0
    for item in iterator:
        feeding_queues[cyclic_idx].put(item)
        cyclic_idx = (cyclic_idx + 1) % len(feeding_queues)


def get_dataset_iterator(dataset, start, stop):
    for idx in range(start, stop):
        yield dataset[idx]


def sequential_processing(args):
    dataset = ByteDataset(data_path=args.bytedataset_path, idx_record_size=6)
    start, stop = args.portion.split(":")
    start = int(start)
    stop = int(stop)
    if stop == -1:
        stop = len(dataset)
    stop = min(len(dataset), stop)
    portion_size = stop - start
    iterator = get_dataset_iterator(dataset, start, stop)

    cache_path = os.path.join(args.bytedataset_path, "cached[{}]".format(args.portion))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    idxs_writer = open(os.path.join(cache_path, "idxs.pkl"), "wb")
    dataset_size = len(dataset)
    idxs_writer.write(dataset_size.to_bytes(4, 'big', signed=False))

    data_writer = open(os.path.join(cache_path, "data.pkl"), "wb")
    for item in tqdm(iterator, desc="Item", total=portion_size):
        out_item = tokenize_item(item)
        idxs_writer.write(data_writer.tell().to_bytes(6, 'big', signed=False))
        pickle.dump(out_item, data_writer)
    
    idxs_writer.close()
    data_writer.close()


def parallel_processing(args):
    dataset = ByteDataset(data_path=args.bytedataset_path, idx_record_size=6)
    start, stop = args.portion.split(":")
    start = int(start)
    stop = int(stop)
    if stop == -1:
        stop = len(dataset)
    stop = min(stop, len(dataset))
    portion_size = stop - start

    cache_path = os.path.join(args.bytedataset_path, "cached[{}]".format(args.portion))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    idxs_writer = open(os.path.join(cache_path, "idxs.pkl"), "wb")
    dataset_size = portion_size
    idxs_writer.write(dataset_size.to_bytes(4, 'big', signed=False))

    data_writer = open(os.path.join(cache_path, "data.pkl"), "wb")

    feeding_queues = [mp.Queue() for _ in range(args.num_processes)]
    fetching_queues = [mp.Queue() for _ in range(args.num_processes)]
    workers = []
    for idx in range(args.num_processes):
        workers.append(mp.Process(target=worker_func, args=(feeding_queues[idx], fetching_queues[idx]), daemon=True))

    feeding_worker = mp.Process(target=feeding, args=(feeding_queues, args), daemon=True)
    feeding_worker.start()

    for worker in workers:
        worker.start()
    
    cyclic_idx = 0
    num_done = 0
    progress_bar = tqdm(total=portion_size, desc="Item")
    while num_done < portion_size:
        out_item = fetching_queues[cyclic_idx].get()
        idxs_writer.write(data_writer.tell().to_bytes(6, 'big', signed=False))
        pickle.dump(out_item, data_writer)
        progress_bar.update(1)
    
    idxs_writer.close()
    data_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytedataset-path", required=True)
    parser.add_argument("--tokenizer-path", default="bert-base-uncased")
    parser.add_argument("--tokenizer-type", choices=["bert"], default="bert")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--portion", default="0:-1")
    args = parser.parse_args()

    tokenizer_class = TOKENIZER_MAPPINGS[args.tokenizer_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    setattr(tokenize_item, 'tokenizer', tokenizer)

    if args.num_processes == 1:
        sequential_processing(args)
    else:
        parallel_processing(args)


if __name__ == "__main__":
    main()

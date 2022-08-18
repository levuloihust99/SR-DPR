import re
import os
import pickle
import argparse
import logging

from dpr.indexer.faiss_indexers import DenseFlatIndexer
from dpr.utils.logging_utils import add_color_formater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def build_index(encoded_ctx_dir: str, indexer: DenseFlatIndexer, index_path: str):
    files = os.listdir(encoded_ctx_dir)
    files = sorted(files, key=lambda x: int(re.search(r"wikipedia_passages_(\d+)\.pkl", x).group(1)))
    files = [os.path.join(encoded_ctx_dir, f) for f in files]
    logger.info("Iterating through encoded context files...")
    for f in files:
        with open(f, "rb") as reader:
            chunk = pickle.load(reader)
            indexer.index_iterated_data(chunk)
        del chunk
    indexer.serialize(index_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--encoded-ctx-dir", required=True)
    args = parser.parse_args()

    indexer = DenseFlatIndexer(vector_sz=768, buffer_size=50000)
    build_index(encoded_ctx_dir=args.encoded_ctx_dir, indexer=indexer, index_path=args.index_path)


if __name__ == "__main__":
    main()

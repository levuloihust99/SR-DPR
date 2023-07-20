"""
 FAISS-based index components for dense retriever
"""

import faiss
import logging
import numpy as np
import os
import pickle
# import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Dict, Text

logger = logging.getLogger()


class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.meta = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[Dict, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[Dict], List[float]]]:
        raise NotImplementedError

    def serialize(self, path: Text):
        logger.info("Serializing index to {}...".format(path))
        if not Path(path).exists():
            os.makedirs(path)
        if Path(path).is_dir():
            index_path = os.path.join(path, "index.dpr")
            meta_path = os.path.join(path, "index_meta.dpr")
        else:
            index_path = path + ".{}.dpr".format(self.get_index_name())
            meta_path = path + ".{}_meta.dpr".format(self.get_index_name())

        with open(index_path, "wb") as write_stream:
            faiss_wrapper_write_stream = faiss.PyCallbackIOWriter(write_stream.write)
            faiss.write_index(self.index, faiss_wrapper_write_stream)
        with open(meta_path, "wb") as write_stream:
            pickle.dump(self.meta, write_stream)

    def get_files(self, path: Text):
        if Path(path).is_dir():
            index_path = os.path.join(path, "index.dpr")
            meta_path = os.path.join(path, "index_meta.dpr")
        else:
            index_path = path + ".{}.dpr".format(self.get_index_name())
            meta_path = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_path, meta_path

    def index_exists(self, path: str):
        index_path, meta_path = self.get_files(path)
        return Path(index_path).exists() and Path(meta_path).exists()

    def deserialize(self, path: Text):
        logger.info("Loading index from {}...".format(path))
        index_path, meta_path = self.get_files(path)

        with open(index_path, "rb") as read_stream:
            faiss_wrapper_read_stream = faiss.PyCallbackIOReader(read_stream.read)
            self.index = faiss.read_index(faiss_wrapper_read_stream)
        logger.info("Loaded index of type `%s` and size %d",
                    type(self.index), self.index.ntotal)

        with open(meta_path, "rb") as read_stream:
            self.meta = pickle.load(read_stream)
        assert (
            len(self.meta) == self.index.ntotal
        ), "Deserialized meta should match faiss index size"

    def _update_meta(self, meta_data: List) -> int:
        self.meta.extend(meta_data)
        return len(self.meta)


class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[Dict, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            meta_data = [t[0] for t in data[i: i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1))
                       for t in data[i: i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_meta(meta_data)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.meta)
        logger.info("Total data indexed {}".format(indexed_cnt))

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        meta_data = [[self.meta[i] for i in query_top_idxs]
                     for query_top_idxs in indexes]
        result = [(meta_data[i], scores[i]) for i in range(len(meta_data))]
        return result

    def get_index_name(self):
        return "inner_product_index"

import os
import json
import pickle
import argparse
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


def process(args):
    idx_writer = open(os.path.join(args.output_bytedataset_path, "idxs.pkl"), "wb")
    dataset_size_place_holder = (0).to_bytes(4, 'big', signed=False)
    idx_writer.write(dataset_size_place_holder)

    data_writer = open(os.path.join(args.output_bytedataset_path, "data.pkl"), "wb")

    num_records = 0
    with open(args.input_jsonl_path) as reader:
        for line in tqdm(reader):
            item = json.loads(line.strip())
            idx_writer.write(data_writer.tell().to_bytes(6, 'big', signed=False))
            pickle.dump(item, data_writer)
            num_records += 1
        
    data_writer.close()
        
    idx_writer.seek(0, 0)
    idx_writer.write(num_records.to_bytes(4, 'big', signed=False))
    idx_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl-path", required=True)
    parser.add_argument("--output-bytedataset-path", required=True)
    parser.add_argument("--log-level", default="debug")
    args = parser.parse_args()

    # setup logging
    log_level = args.log_level
    if log_level is None:
        log_level = 'info'
    log_level = log_level.upper()
    if not hasattr(logging, log_level):
        raise Exception("The log level '{}' is invalid.".format(args.log_level))
    logging.basicConfig(level=getattr(logging, log_level))

    process(args)


if __name__ == "__main__":
    main()

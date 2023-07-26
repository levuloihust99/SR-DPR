import logging
import time
# import tensorflow as tf
from typing import Text
import torch 
from transformers import BertModel

logger = logging.getLogger()

def recursive_apply(data, fn, ignore_keys=None):
    """Hàm áp dụng hồi quy function {fn} vào data

    Args:
        data (Dict/List): _description_
        fn (function): _description_
        ignore_keys (_type_, optional): Key của Dict không áp dụng function. Defaults to None.

    Returns:
        data: _description_
    """    
    stack = [(None, -1, data)]  # parent, idx, child: parent[idx] = child
    while stack:
        parent_node, index, node = stack.pop()
        if isinstance(node, list):
            stack.extend(list(zip([node] * len(node), range(len(node)), node)))
        elif isinstance(node, dict):
            stack.extend(
                list(zip([node] * len(node), node.keys(), node.values())))
        elif isinstance(node, str):
            if node and (ignore_keys is None or index not in ignore_keys):
                parent_node[index] = fn(node)
            else:
                parent_node[index] = node
        else:
            continue
    return data

def load_query_encoder(checkpoint_path: Text, pretrained_model_path: Text):
    logger.info("Loading query encoder...")
    start_time = time.perf_counter()
    
    # encoder_class = MODEL_MAPPING[architecture]
    saved_state = torch.load(checkpoint_path, map_location=lambda s, t: s)
    saved_state = saved_state['model_dict']
    question_prefix = "question_model."
    
    # Question Model
    question_model_state = {
            k[len(question_prefix):]: v
            for k, v in saved_state.items()
            if k.startswith(question_prefix)
        }
    # assert len(question_model_state) == 199s
    question_model = BertModel.from_pretrained(pretrained_model_path)
    question_model.load_state_dict(question_model_state, strict= False)
    
    logger.info("Done loading query encoder in {}s".format(
        time.perf_counter() - start_time))
    # return question_model, ctx_model
    return question_model
    
class ColorFormatter(logging.Formatter):
    BOLD_GREEN = "\x1b[38;5;2;1m"
    RESET_CODE = "\x1b[0m"
    FORMAT_STR = BOLD_GREEN + "%(asctime)s - %(name)s - %(levelname)s: " + RESET_CODE + "%(message)s"
    
    def format(self, record):
        formatter = logging.Formatter(self.FORMAT_STR)
        return formatter.format(record)

def add_color_formater(logger: logging.Logger):
    handlers = logger.handlers
    for handler in handlers:
        handler.setFormatter(ColorFormatter())
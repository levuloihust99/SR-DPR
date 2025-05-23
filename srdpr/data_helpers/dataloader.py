import copy
import random
import multiprocessing as mp
import torch
import logging

from collections import deque
from typing import Optional
from torch.utils.data import IterableDataset, DataLoader

from dpr.utils.data_utils import normalize_question
from srdpr.data_helpers.datasets import ByteDataset
from srdpr.utils.helpers import recursive_apply
from transformers import BertTokenizer


class StatelessIdxsGenerator(object):
    def __init__(self, N, shuffle_seed: int = 0, epoch: int = 0, iteration: int = 0):
        self.sequence = list(range(N))
        self.shuffle_seed = shuffle_seed
        self.epoch = epoch
        self.iteration = iteration
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_iteration(self, iteration):
        self.iteration = iteration

    def __iter__(self):
        while True:
            idxs = copy.copy(self.sequence)
            rnd = random.Random(self.shuffle_seed + self.epoch)
            rnd.shuffle(idxs)
            try:
                self.iteration += 1
                yield idxs[self.iteration - 1]
            except IndexError:
                self.epoch += 1
                self.iteration = 0


class PosDataIterator(object):
    def __init__(self,
        dataset: ByteDataset,
        idxs_generator: StatelessIdxsGenerator,
        tokenizer,
        forward_batch_size: int,
        contrastive_size: int,
        prefetch_factor: int = 10,
        max_length: int = 512,
        shuffle_positive: bool = False,
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.idxs_generator = idxs_generator
        self.tokenizer = tokenizer
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.max_length = max_length
        self.shuffle_positive = shuffle_positive
        self.shuffle = shuffle
        self.state = {'shift_back': 0}
        self.data_queue = mp.Queue(maxsize=prefetch_factor)
        self.feeding_worker = mp.Process(target=self.feeding, 
            args=(self.data_queue, self.dataset.data_path, self.idxs_generator, self.collate_fn,
                self.forward_batch_size, self.contrastive_size), daemon=True)
        self.buffer = deque(maxlen=forward_batch_size + contrastive_size)

    def start_worker(self):
        logging.getLogger("transformers.tokenization_utils_base").disabled = True
        self.feeding_worker.start()

    def get_idxs_generator_state(self):
        return {'epoch': self.idxs_generator.epoch, 'iteration': self.idxs_generator.iteration,
            'shift_back': self.state['shift_back']}
    
    def set_idxs_generator_state(self, state):
        self.idxs_generator.set_epoch(state['epoch'])
        self.idxs_generator.set_iteration(state['iteration'] - state['shift_back'])
    
    def __iter__(self):
        return self

    def __next__(self):
        batch, update = self.data_queue.get()
        self.idxs_generator.set_epoch(update['epoch'])
        self.idxs_generator.set_iteration(update['iteration'])
        self.state['shift_back'] = update['shift_back']
        return batch
    
    @staticmethod
    def feeding(data_queue, dataset_path, idxs_generator, collate_fn, forward_batch_size, contrastive_size):
        dataset = ByteDataset(data_path=dataset_path, idx_record_size=6)
        idxs_iterator = iter(idxs_generator)
        buffer = deque(maxlen=forward_batch_size + contrastive_size)
        while True:
            # fill up buffer
            while len(buffer) < forward_batch_size + contrastive_size:
                buffer.append(next(idxs_iterator))
            
            # shift buffer
            forward_idxs = []
            for _ in range(forward_batch_size):
                forward_idxs.append(buffer.popleft())
            
            # access buffer
            contrastive_idxs = []
            for i in range(contrastive_size):
                contrastive_idxs.append(buffer[i])
            
            idxs = forward_idxs + contrastive_idxs
            items = [dataset[idx] for idx in idxs]
            if collate_fn:
                random.seed(idxs_generator.shuffle_seed + idxs_generator.epoch
                    + idxs_generator.iteration)
                items = collate_fn(items)
            data_queue.put((items, {'epoch': idxs_generator.epoch, 'iteration': idxs_generator.iteration,
                'shift_back': len(buffer)}))

    def collate_fn(self, items):
        """This function use dynamic batching."""

        batch_questions_ids = []
        batch_positive_contexts_ids = []
        batch_negative_contexts_ids = []

        max_batch_question_len = 0
        max_batch_context_len = 0
        for idx, item in enumerate(items):
            if idx < self.forward_batch_size:
                questions = copy.copy(item['questions'])
                random.shuffle(questions)
                question = questions[0]
                question = normalize_question(question)
                question = question.strip()
                question_ids = self.tokenizer.encode(question, add_special_tokens=True)
                if len(question_ids) > self.max_length:
                    question_ids = question_ids[:self.max_length]
                    question_ids[-1] = self.tokenizer.sep_token_id
                if max_batch_question_len < len(question_ids):
                    max_batch_question_len = len(question_ids)
                batch_questions_ids.append(question_ids)

                positive_contexts = copy.copy(item['positive_contexts'])
                if self.shuffle_positive:
                    random.shuffle(positive_contexts)
                positive_context = copy.copy(positive_contexts[0])
                recursive_apply(positive_context, fn=lambda x: x.strip())
                positive_context_ids = self.tokenizer.encode(positive_context['title'],
                    text_pair=positive_context['text'], add_special_tokens=True)
                if len(positive_context_ids) > self.max_length:
                    positive_context_ids = positive_context_ids[:self.max_length]
                    positive_context_ids[-1] = self.tokenizer.sep_token_id
                if max_batch_context_len < len(positive_context_ids):
                    max_batch_context_len = len(positive_context_ids)
                batch_positive_contexts_ids.append(positive_context_ids)
            else:
                negative_contexts = copy.copy(item['positive_contexts'] + item['hardneg_contexts'])
                if self.shuffle:
                    random.shuffle(negative_contexts)
                negative_context = copy.copy(negative_contexts[0])
                recursive_apply(negative_context, fn=lambda x: x.strip())
                negative_context_ids = self.tokenizer.encode(negative_context['title'],
                    text_pair=negative_context['text'], add_special_tokens=True)
                if len(negative_context_ids) > self.max_length:
                    negative_context_ids = negative_context_ids[:self.max_length]
                    negative_context_ids[-1] = self.tokenizer.sep_token_id
                if max_batch_context_len < len(negative_context_ids):
                    max_batch_context_len = len(negative_context_ids)
                batch_negative_contexts_ids.append(negative_context_ids)
        
        batch = {
            'question/input_ids': [],
            'question/attn_mask': [],
            'positive_context/input_ids': [],
            'positive_context/attn_mask': [],
            'negative_context/input_ids': [],
            'negative_context/attn_mask': []
        }

        for idx, question_ids in enumerate(batch_questions_ids):
            padding_len = max_batch_question_len - len(question_ids)
            q_attn_mask = [1] * len(question_ids) + [0] * padding_len
            question_ids += [self.tokenizer.pad_token_id] * padding_len
            batch['question/input_ids'].append(question_ids)
            batch['question/attn_mask'].append(q_attn_mask)
        
        for idx, positive_context_ids in enumerate(batch_positive_contexts_ids):
            padding_len = max_batch_context_len - len(positive_context_ids)
            c_attn_mask = [1] * len(positive_context_ids) + [0] * padding_len
            positive_context_ids += [self.tokenizer.pad_token_id] * padding_len
            batch['positive_context/input_ids'].append(positive_context_ids)
            batch['positive_context/attn_mask'].append(c_attn_mask)
        
        for idx, negative_context_ids in enumerate(batch_negative_contexts_ids):
            padding_len = max_batch_context_len - len(negative_context_ids)
            c_attn_mask = [1] * len(negative_context_ids) + [0] * padding_len
            negative_context_ids += [self.tokenizer.pad_token_id] * padding_len
            batch['negative_context/input_ids'].append(negative_context_ids)
            batch['negative_context/attn_mask'].append(c_attn_mask)
        
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        # compute duplicate mask
        ids = torch.tensor([item['sample_id'] for item in items])
        batch['ids'] = ids
        positive_ids = ids[:self.forward_batch_size]
        negative_ids = ids[self.forward_batch_size:]
        duplicate_mask = positive_ids.unsqueeze(1).repeat(1, self.contrastive_size) != \
                negative_ids.unsqueeze(0).repeat(self.forward_batch_size, 1)
        duplicate_mask = duplicate_mask.to(torch.int64)
        duplicate_mask = torch.cat([torch.ones(self.forward_batch_size, 1, dtype=torch.int64), duplicate_mask], dim=1)
        batch['duplicate_mask'] = duplicate_mask
        return batch


class PoshardDataIterator(object):
    def __init__(
        self,
        dataset: ByteDataset,
        idxs_generator: StatelessIdxsGenerator,
        tokenizer,
        forward_batch_size: int,
        contrastive_size: int,
        limit_hardnegs: int = -1,
        prefetch_factor: int = 10,
        shuffle_positive: bool = False,
        shuffle: bool = True,
        max_length: int =  512
    ):
        self.dataset = dataset
        self.idxs_generator = idxs_generator
        self.tokenizer = tokenizer
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.limit_hardnegs = limit_hardnegs
        self.shuffle_positive = shuffle_positive
        self.shuffle = shuffle
        self.max_length = max_length
        self.data_queue = mp.Queue(maxsize=prefetch_factor)
        self.feeding_worker = mp.Process(target=self.feeding, args=(self.data_queue, dataset.data_path,
            idxs_generator, forward_batch_size, self.collate_fn), daemon=True)
    
    def start_worker(self):
        logging.getLogger("transformers.tokenization_utils_base").disabled = True
        self.feeding_worker.start()
    
    def get_idxs_generator_state(self):
        return {'epoch': self.idxs_generator.epoch, 'iteration': self.idxs_generator.iteration}
    
    def set_idxs_generator_state(self, state):
        self.idxs_generator.set_epoch(state['epoch'])
        self.idxs_generator.set_iteration(state['iteration'])

    def __iter__(self):
        return self

    def __next__(self):
        batch, update = self.data_queue.get()
        self.idxs_generator.set_epoch(update['epoch'])
        self.idxs_generator.set_iteration(update['iteration'])
        return batch

    @staticmethod
    def feeding(
        data_queue,
        data_path,
        idxs_generator,
        forward_batch_size,
        collate_fn
    ):
        idxs_iterator = iter(idxs_generator)
        dataset = ByteDataset(data_path=data_path, idx_record_size=6)
        while True:
            idxs = [next(idxs_iterator) for _ in range(forward_batch_size)]
            items = [dataset[idx] for idx in idxs]
            if collate_fn:
                random.seed(idxs_generator.shuffle_seed + idxs_generator.epoch + idxs_generator.iteration)
                items = collate_fn(items)
            data_queue.put((items, {'epoch': idxs_generator.epoch, 'iteration': idxs_generator.iteration}))

    def collate_fn(self, items):
        """This function performs dynamic contrastive size."""

        dynamic_contrastive_size = 0
        batch_questions = []
        batch_positive_contexts = []
        batch_hardneg_contexts = []

        # sample and padding
        for item in items:
            questions = copy.copy(item['questions'])
            random.shuffle(questions)
            question = questions[0]
            batch_questions.append(question)

            positive_contexts = copy.copy(item['positive_contexts'])
            if self.shuffle_positive:
                random.shuffle(positive_contexts)
            positive_context = copy.copy(positive_contexts[0])
            batch_positive_contexts.append(positive_context)

            hardneg_contexts = copy.copy(item['hardneg_contexts'])
            if self.limit_hardnegs > 0:
                hardneg_contexts = hardneg_contexts[:self.limit_hardnegs]
            if self.shuffle:
                random.shuffle(hardneg_contexts)
            if dynamic_contrastive_size < len(hardneg_contexts):
                dynamic_contrastive_size = len(hardneg_contexts)
            batch_hardneg_contexts.append(hardneg_contexts)
        
        contrastive_size = min(dynamic_contrastive_size, self.contrastive_size)
        hardneg_padding_lens = []
        batch_hardneg_mask = []
        for idx, hardneg_contexts in enumerate(batch_hardneg_contexts):
            batch_hardneg_contexts[idx] = hardneg_contexts[:contrastive_size]
            hardneg_contexts = batch_hardneg_contexts[idx]
            hardneg_mask = [1] * len(hardneg_contexts)
            padding_len = contrastive_size - len(hardneg_contexts)
            hardneg_padding_lens.append(padding_len)
            if padding_len > 0:
                hardneg_mask += [0] * padding_len
            batch_hardneg_mask.append(hardneg_mask)
        batch_hardneg_mask = torch.tensor(batch_hardneg_mask)
        batch_size = batch_hardneg_mask.size(0)
        batch_hardneg_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.long),
            batch_hardneg_mask], dim=1)
        
        # tokenize
        max_batch_question_len = 0
        batch_question_ids = []
        for question in batch_questions:
            question = normalize_question(question)
            question = question.strip()
            question_ids = self.tokenizer.encode(question, add_special_tokens=True)
            if len(question_ids) > self.max_length:
                question_ids = question_ids[:self.max_length]
                question_ids[-1] = self.tokenizer.sep_token_id
            batch_question_ids.append(question_ids)
            if max_batch_question_len < len(question_ids):
                max_batch_question_len = len(question_ids)

        batch_question_tensors = []
        batch_question_attn_mask = []
        for question_ids in batch_question_ids:
            q_attn_mask = [1] * len(question_ids)
            padding_len = max_batch_question_len - len(question_ids)
            if padding_len > 0:
                question_ids += [self.tokenizer.pad_token_id] * padding_len
                q_attn_mask += [0] * padding_len
            batch_question_tensors.append(question_ids)
            batch_question_attn_mask.append(q_attn_mask)
        batch_question_tensors = torch.tensor(batch_question_tensors)
        batch_question_attn_mask = torch.tensor(batch_question_attn_mask)

        max_batch_context_len = 0
        batch_positive_context_ids = []
        for positive_context in batch_positive_contexts:
            recursive_apply(positive_context, fn=lambda x: x.strip())
            positive_context_ids = self.tokenizer.encode(positive_context['title'],
                    text_pair=positive_context['text'], add_special_tokens=True)
            if len(positive_context_ids) > self.max_length:
                positive_context_ids = positive_context_ids[:self.max_length]
                positive_context_ids[-1] = self.tokenizer.sep_token_id
            if max_batch_context_len < len(positive_context_ids):
                max_batch_context_len = len(positive_context_ids)
            batch_positive_context_ids.append(positive_context_ids)
        
        batch_hardnegs_context_ids = []
        for hardneg_contexts in batch_hardneg_contexts:
            hardneg_contexts = [copy.copy(c) for c in hardneg_contexts]
            recursive_apply(hardneg_contexts, fn=lambda x: x.strip())
            per_sample_hardnegs_context_ids = []
            for hardneg_context in hardneg_contexts:
                hardneg_context_ids = self.tokenizer.encode(hardneg_context['title'],
                    text_pair=hardneg_context['text'], add_special_tokens=True)
                if len(hardneg_context_ids) > self.max_length:
                    hardneg_context_ids = hardneg_context_ids[:self.max_length]
                    hardneg_context_ids[-1] = self.tokenizer.sep_token_id
                per_sample_hardnegs_context_ids.append(hardneg_context_ids)
                if max_batch_context_len < len(hardneg_context_ids):
                    max_batch_context_len = len(hardneg_context_ids)
            batch_hardnegs_context_ids.append(per_sample_hardnegs_context_ids)
        
        batch_positive_context_tensors = []
        batch_positive_context_attn_mask = []
        for positive_context_ids in batch_positive_context_ids:
            c_attn_mask = [1] * len(positive_context_ids)
            padding_len = max_batch_context_len - len(positive_context_ids)
            if padding_len > 0:
                positive_context_ids += [self.tokenizer.pad_token_id] * padding_len
                c_attn_mask += [0] * padding_len
            batch_positive_context_tensors.append(positive_context_ids)
            batch_positive_context_attn_mask.append(c_attn_mask)
        batch_positive_context_tensors = torch.tensor(batch_positive_context_tensors)
        batch_positive_context_attn_mask = torch.tensor(batch_positive_context_attn_mask)

        batch_hardnegs_context_tensors = []
        batch_hardnegs_context_attn_mask = []
        for idx, hardnegs_context_ids in enumerate(batch_hardnegs_context_ids):
            per_sample_hardneg_contexts_tensors = []
            per_sample_hardneg_contexts_attn_mask = []
            for hardneg_context_ids in hardnegs_context_ids:
                c_attn_mask = [1] * len(hardneg_context_ids)
                padding_len = max_batch_context_len - len(hardneg_context_ids)
                if padding_len > 0:
                    hardneg_context_ids += [self.tokenizer.pad_token_id] * padding_len
                    c_attn_mask += [0] * padding_len
                per_sample_hardneg_contexts_tensors.append(hardneg_context_ids)
                per_sample_hardneg_contexts_attn_mask.append(c_attn_mask)
            per_sample_hardneg_contexts_tensors = torch.tensor(per_sample_hardneg_contexts_tensors)
            per_sample_hardneg_contexts_attn_mask = torch.tensor(per_sample_hardneg_contexts_attn_mask)
            if hardneg_padding_lens[idx] > 0:
                per_sample_hardneg_contexts_tensors = torch.cat(
                    [per_sample_hardneg_contexts_tensors,
                    torch.zeros(hardneg_padding_lens[idx], max_batch_context_len, dtype=torch.long)],
                    dim=0
                )
                per_sample_hardneg_contexts_attn_mask = torch.cat(
                    [per_sample_hardneg_contexts_attn_mask,
                    torch.zeros(hardneg_padding_lens[idx], max_batch_context_len, dtype=torch.long)],
                    dim=0
                )
            batch_hardnegs_context_tensors.append(per_sample_hardneg_contexts_tensors)
            batch_hardnegs_context_attn_mask.append(per_sample_hardneg_contexts_attn_mask)
        batch_hardnegs_context_tensors = torch.stack(batch_hardnegs_context_tensors, dim=0)
        batch_hardnegs_context_attn_mask = torch.stack(batch_hardnegs_context_attn_mask, dim=0)

        ids = torch.tensor([item['sample_id'] for item in items])
        
        return {
            'question/input_ids': batch_question_tensors,
            'question/attn_mask': batch_question_attn_mask,
            'positive_context/input_ids': batch_positive_context_tensors,
            'positive_context/attn_mask': batch_positive_context_attn_mask,
            'hardneg_context/input_ids': batch_hardnegs_context_tensors,
            'hardneg_context/attn_mask': batch_hardnegs_context_attn_mask,
            'hardneg_mask': batch_hardneg_mask,
            'ids': ids
        }


class HardDataIterator(object):
    def __init__(
        self,
        hardneg_dataset: ByteDataset,
        hardneg_idxs_generator: StatelessIdxsGenerator,
        tokenizer,
        forward_batch_size: int,
        contrastive_size: int,
        randneg_dataset: Optional[ByteDataset] = None,
        randneg_idxs_generator: Optional[StatelessIdxsGenerator] = None,
        shuffle: bool = True,
        max_length: int = 512,
        use_randneg_dataset: bool = False,
        prefetch_factor: int = 10,
    ):
        self.hardneg_dataset = hardneg_dataset
        self.hardneg_idxs_generator = hardneg_idxs_generator
        self.tokenizer = tokenizer
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.use_randneg_dataset = use_randneg_dataset
        self.randneg_dataset = randneg_dataset
        self.randneg_idxs_generator = randneg_idxs_generator
        self.state = {'hardneg': {'shift_back': 0}}
        
        self.data_queue = mp.Queue(maxsize=prefetch_factor)
        self.sample_from_hardneg = None
        if use_randneg_dataset:
            self._determine_sample_size()
        self.feeding_worker = mp.Process(target=self.feeding, 
            args=(self.data_queue, hardneg_dataset.data_path, hardneg_idxs_generator, self.collate_fn, forward_batch_size,
                contrastive_size, randneg_dataset.data_path if randneg_dataset else None, randneg_idxs_generator, self.sample_from_hardneg, use_randneg_dataset),
            daemon=True)

    def _determine_sample_size(self):
        hardneg_dataset_size = len(self.hardneg_dataset)
        randneg_dataset_size = len(self.randneg_dataset)
        sample_from_hardneg = self.contrastive_size * hardneg_dataset_size / (randneg_dataset_size + hardneg_dataset_size)
        self.sample_from_hardneg = int(sample_from_hardneg)
    
    def start_worker(self):
        logging.getLogger("transformers.tokenization_utils_base").disabled = True
        self.feeding_worker.start()
    
    def get_idxs_generator_state(self):
        state = {
            'hardneg': {
                'epoch': self.hardneg_idxs_generator.epoch,
                'iteration': self.hardneg_idxs_generator.iteration,
                'shift_back': self.state['hardneg']['shift_back']
            }
        }
        if self.use_randneg_dataset:
            state['randneg'] = {
                'epoch': self.randneg_idxs_generator.epoch,
                'iteration': self.randneg_idxs_generator.iteration
            }
        return state
    
    def set_idxs_generator_state(self, state):
        hardneg_state = state['hardneg']
        self.hardneg_idxs_generator.set_epoch(hardneg_state['epoch'])
        self.hardneg_idxs_generator.set_iteration(hardneg_state['iteration'] - hardneg_state['shift_back'])
        if self.use_randneg_dataset:
            randneg_state = state['randneg']
            self.randneg_idxs_generator.set_epoch(randneg_state['epoch'])
            self.randneg_idxs_generator.set_iteration(randneg_state['iteration'])
    
    def __iter__(self):
        return self

    def __next__(self):
        batch, update = self.data_queue.get()
        self.hardneg_idxs_generator.set_epoch(update['hardneg']['epoch'])
        self.hardneg_idxs_generator.set_iteration(update['hardneg']['iteration'])
        if self.use_randneg_dataset:
            self.randneg_idxs_generator.set_epoch(update['randneg']['epoch'])
            self.randneg_idxs_generator.set_iteration(update['randneg']['iteration'])
        self.state['hardneg']['shift_back'] = update['hardneg']['shift_back']
        return batch

    @staticmethod
    def feeding(
        data_queue,
        hardneg_dataset_path,
        hardneg_idxs_generator,
        collate_fn,
        forward_batch_size,
        contrastive_size,
        randneg_dataset_path,
        randneg_idxs_generator,
        sample_from_hardneg,
        use_randneg_dataset
    ):
        hardneg_dataset = ByteDataset(data_path=hardneg_dataset_path, idx_record_size=6)
        if not use_randneg_dataset:
            hardneg_idxs_iterator = iter(hardneg_idxs_generator)
            buffer = deque(maxlen=forward_batch_size + contrastive_size)
            while True:
                # fill up buffer
                while len(buffer) < forward_batch_size + contrastive_size:
                    buffer.append(next(hardneg_idxs_iterator))
                
                # shift buffer
                forward_idxs = []
                for _ in range(forward_batch_size):
                    forward_idxs.append(buffer.popleft())
                
                # access buffer
                contrastive_idxs = []
                for i in range(contrastive_size):
                    contrastive_idxs.append(buffer[i])
                
                idxs = forward_idxs + contrastive_idxs
                items = [hardneg_dataset[idx] for idx in idxs]
                if collate_fn:
                    random.seed(hardneg_idxs_generator.shuffle_seed + hardneg_idxs_generator.epoch
                        + hardneg_idxs_generator.iteration)
                    items = collate_fn(items)
                data_queue.put((items, {'hardneg': {'epoch': hardneg_idxs_generator.epoch, 'iteration': hardneg_idxs_generator.iteration,
                    'shift_back': len(buffer)}}))
        else:
            hardneg_idxs_iterator = iter(hardneg_idxs_generator)
            randneg_dataset = ByteDataset(data_path=randneg_dataset_path, idx_record_size=6)
            randneg_idxs_iterator = iter(randneg_idxs_generator)
            buffer = deque(maxlen=forward_batch_size + sample_from_hardneg)
            while True:
                # fill up buffer
                while len(buffer) < forward_batch_size + sample_from_hardneg:
                    buffer.append(next(hardneg_idxs_iterator))
                
                # shift buffer
                forward_idxs = []
                for _ in range(forward_batch_size):
                    forward_idxs.append(buffer.popleft())
                
                # access buffer
                hardneg_contrastive_idxs = []
                for i in range(sample_from_hardneg):
                    hardneg_contrastive_idxs.append(buffer[i])
                randneg_contrastive_idxs = []
                for _ in range(contrastive_size - sample_from_hardneg):
                    randneg_contrastive_idxs.append(next(randneg_idxs_iterator))
                
                items = [hardneg_dataset[idx] for idx in forward_idxs + hardneg_contrastive_idxs] + \
                            [randneg_dataset[idx] for idx in randneg_contrastive_idxs]

                if collate_fn:
                    random.seed(hardneg_idxs_generator.shuffle_seed + hardneg_idxs_generator.epoch
                        + hardneg_idxs_generator.iteration + randneg_idxs_generator.shuffle_seed
                        + randneg_idxs_generator.epoch + randneg_idxs_generator.iteration)
                    items = collate_fn(items)
                data_queue.put((items, {
                    'hardneg': {
                        'epoch': hardneg_idxs_generator.epoch,
                        'iteration': hardneg_idxs_generator.iteration,
                        'shift_back': len(buffer)
                    },
                    'randneg': {
                        'epoch': randneg_idxs_generator.epoch,
                        'iteration': randneg_idxs_generator.iteration
                    }
                }))

    def collate_fn(self, items):
        """This function use dynamic batching."""
        batch_questions_ids = []
        batch_hardneg_contexts_ids = []
        batch_randneg_contexts_ids = []

        max_batch_question_len = 0
        max_batch_context_len = 0
        for idx, item in enumerate(items):
            if idx < self.forward_batch_size:
                questions = copy.copy(item['questions'])
                random.shuffle(questions)
                question = questions[0]
                question = normalize_question(question)
                question = question.strip()
                question_ids = self.tokenizer.encode(question, add_special_tokens=True)
                if len(question_ids) > self.max_length:
                    question_ids = question_ids[:self.max_length]
                    question_ids[-1] = self.tokenizer.sep_token_id
                if max_batch_question_len < len(question_ids):
                    max_batch_question_len = len(question_ids)
                batch_questions_ids.append(question_ids)

                hardneg_contexts = copy.copy(item['hardneg_contexts'])
                if self.shuffle:
                    random.shuffle(hardneg_contexts)
                hardneg_context = copy.copy(hardneg_contexts[0])
                recursive_apply(hardneg_context, fn=lambda x: x.strip())
                hardneg_context_ids = self.tokenizer.encode(hardneg_context['title'],
                    text_pair=hardneg_context['text'], add_special_tokens=True)
                if len(hardneg_context_ids) > self.max_length:
                    hardneg_context_ids = hardneg_context_ids[:self.max_length]
                    hardneg_context_ids[-1] = self.tokenizer.sep_token_id
                if max_batch_context_len < len(hardneg_context_ids):
                    max_batch_context_len = len(hardneg_context_ids)
                batch_hardneg_contexts_ids.append(hardneg_context_ids)
            else:
                randneg_contexts = copy.copy(item['positive_contexts'] + item['hardneg_contexts'])
                if self.shuffle:
                    random.shuffle(randneg_contexts)
                randneg_context = copy.copy(randneg_contexts[0])
                recursive_apply(randneg_context, fn=lambda x: x.strip())
                randneg_context_ids = self.tokenizer.encode(randneg_context['title'],
                    text_pair=randneg_context['text'], add_special_tokens=True)
                if len(randneg_context_ids) > self.max_length:
                    randneg_context_ids = randneg_context_ids[:self.max_length]
                    randneg_context_ids[-1] = self.tokenizer.sep_token_id
                if max_batch_context_len < len(randneg_context_ids):
                    max_batch_context_len = len(randneg_context_ids)
                batch_randneg_contexts_ids.append(randneg_context_ids)
        
        batch = {
            'question/input_ids': [],
            'question/attn_mask': [],
            'positive_context/input_ids': [],
            'positive_context/attn_mask': [],
            'negative_context/input_ids': [],
            'negative_context/attn_mask': []
        }

        for idx, question_ids in enumerate(batch_questions_ids):
            padding_len = max_batch_question_len - len(question_ids)
            q_attn_mask = [1] * len(question_ids) + [0] * padding_len
            question_ids += [self.tokenizer.pad_token_id] * padding_len
            batch['question/input_ids'].append(question_ids)
            batch['question/attn_mask'].append(q_attn_mask)
        
        for idx, hardneg_context_ids in enumerate(batch_hardneg_contexts_ids):
            padding_len = max_batch_context_len - len(hardneg_context_ids)
            c_attn_mask = [1] * len(hardneg_context_ids) + [0] * padding_len
            hardneg_context_ids += [self.tokenizer.pad_token_id] * padding_len
            batch['positive_context/input_ids'].append(hardneg_context_ids) # hard pipeline is the same as pos pipeline
                                                                            # when calculating loss. Hence we use the same dict
                                                                            # keys. Here, `hardneg` acts as 'positive'
                                                                            # while `randneg` acts as 'negative'
            batch['positive_context/attn_mask'].append(c_attn_mask)
        
        for idx, randneg_context_ids in enumerate(batch_randneg_contexts_ids):
            padding_len = max_batch_context_len - len(randneg_context_ids)
            c_attn_mask = [1] * len(randneg_context_ids) + [0] * padding_len
            randneg_context_ids += [self.tokenizer.pad_token_id] * padding_len
            batch['negative_context/input_ids'].append(randneg_context_ids)
            batch['negative_context/attn_mask'].append(c_attn_mask)
        
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        # compute duplicate mask
        ids = torch.tensor([item['sample_id'] for item in items])
        batch['ids'] = ids
        positive_ids = ids[:self.forward_batch_size]
        negative_ids = ids[self.forward_batch_size:]
        duplicate_mask = positive_ids.unsqueeze(1).repeat(1, self.contrastive_size) != \
                negative_ids.unsqueeze(0).repeat(self.forward_batch_size, 1)
        duplicate_mask = duplicate_mask.to(torch.int64)
        duplicate_mask = torch.cat([torch.ones(self.forward_batch_size, 1, dtype=torch.int64), duplicate_mask], dim=1)
        batch['duplicate_mask'] = duplicate_mask
        return batch


class InbatchDataIterator(object):
    def __init__(
        self,
        dataset: ByteDataset,
        idxs_generator: StatelessIdxsGenerator,
        tokenizer,
        forward_batch_size: int,
        use_hardneg: bool = True,
        use_num_hardnegs: int = 1,
        max_length: int = 512,
        shuffle_positive: bool = False,
        shuffle: bool = True,
        prefetch_factor: int = 10
    ):
        self.dataset = dataset
        self.idxs_generator = idxs_generator
        self.tokenizer = tokenizer
        self.forward_batch_size = forward_batch_size
        self.use_hardneg = use_hardneg
        self.use_num_hardnegs = use_num_hardnegs
        self.max_length = max_length
        self.shuffle_positive = shuffle_positive
        self.shuffle = shuffle
        self.data_queue = mp.Queue(maxsize=prefetch_factor)
        self.feeding_worker = mp.Process(target=self.feeding,
            args=(self.data_queue, dataset.data_path, self.idxs_generator,
                self.collate_fn, self.forward_batch_size), daemon=True)

    def start_worker(self):
        logging.getLogger("transformers.tokenization_utils_base").disabled = True
        self.feeding_worker.start()
    
    def get_idxs_generator_state(self):
        return {'epoch': self.idxs_generator.epoch, 'iteration': self.idxs_generator.iteration}
    
    def set_idxs_generator_state(self, epoch, iteration):
        self.idxs_generator.set_epoch(epoch)
        self.idxs_generator.set_iteration(iteration)

    def __iter__(self):
        return self

    def __next__(self):
        batch, update = self.data_queue.get()
        self.idxs_generator.set_epoch(update['epoch'])
        self.idxs_generator.set_iteration(update['iteration'])
        return batch
    
    @staticmethod
    def feeding(
        data_queue,
        data_path,
        idxs_generator: StatelessIdxsGenerator,
        collate_fn,
        forward_batch_size,
    ):
        dataset = ByteDataset(data_path=data_path, idx_record_size=6)
        idxs_iterator = iter(idxs_generator)
        while True:
            idxs = [next(idxs_iterator) for _ in range(forward_batch_size)]
            items = [dataset[idx] for idx in idxs]
            if collate_fn:
                random.seed(idxs_generator.shuffle_seed + idxs_generator.epoch + idxs_generator.iteration)
                items = collate_fn(items)
            data_queue.put((items, {'epoch': idxs_generator.epoch, 'iteration': idxs_generator.iteration}))

    def collate_fn(self, items):
        batch_question_ids = []
        batch_positive_context_ids = []

        max_batch_question_len = 0 # dynamic sequence length of question, always less than self.max_length
        max_batch_context_len = 0 # dynamic sequence length of contexts (both positive and hard negative), always less than self.max_length
        
        if self.use_hardneg:
            batch_hardneg_contexts = []
            dynamic_hardneg_num = 0 # dynamic number of hard negative contexts.

        # this loop is to:
        # 1. tokenize questions and get max_batch_question_len
        # 2. tokenize positive contexts and get max_batch_context_len (not finally)
        # 3. do not tokenize hard negative contexts at this loop since we do not know
        # how many contexts to tokenize. This loop does sampling and determine dynamic
        # number of hard negative contexts.
        # By using shallow copy, items in dataset are preserved, necessary for perfectly resume training
        for item in items:
            questions = copy.copy(item['questions']) # use shallow copy for shuffling not to
                                                     # affect item['questions']
            random.shuffle(questions) # avoid in-place change of item['questions']
            question = questions[0]
            question = normalize_question(question) # remove question mark
            question = question.strip()
            question_ids = self.tokenizer.encode(question, add_special_tokens=True) # explicitly add
                                                            # special tokens for guaranteeing version difference of transformers.
            if len(question_ids) > self.max_length: # truncate inputs that exceed max_length
                question_ids = question_ids[:self.max_length]
                question_ids[-1] = self.tokenizer.sep_token_id
            if max_batch_question_len < len(question_ids): # dynamic max sequence length for this batch
                max_batch_question_len = len(question_ids)
            batch_question_ids.append(question_ids) # 1. get batched tokenized questions.

            positive_contexts = copy.copy(item['positive_contexts']) # use shallow copy for shuffling not to
                                                                     # affect item['positive_contexts']
            if self.shuffle_positive:
                random.shuffle(positive_contexts) # change positive_contexts inplace, that's why we need shallow copy
            positive_context = copy.copy(positive_contexts[0]) # shallow copy for stripping not to
                                                               # affect `positive_contexts`
            recursive_apply(positive_context, lambda x: x.strip()) # change positive_context inplace.
            positive_context_ids = self.tokenizer.encode(positive_context['title'],
                text_pair=positive_context['text'], add_special_tokens=True)
            if len(positive_context_ids) > self.max_length:
                positive_context_ids = positive_context_ids[:self.max_length]
                positive_context_ids[-1] = self.tokenizer.sep_token_id
            if max_batch_context_len < len(positive_context_ids):
                max_batch_context_len = len(positive_context_ids)
            batch_positive_context_ids.append(positive_context_ids)

            if self.use_hardneg:
                hardneg_contexts = copy.copy(item['hardneg_contexts']) # shallow copy for shuffling not to
                                                                    # affect item['hardneg_contexts']
                if self.shuffle:
                    random.shuffle(hardneg_contexts) # change hardneg_contexts inplace.
                hardneg_contexts = hardneg_contexts[:self.use_num_hardnegs] # number of hard negatives cannot
                                                                            # exceeds `use_num_hardnegs`
                if dynamic_hardneg_num < len(hardneg_contexts):
                    dynamic_hardneg_num = len(hardneg_contexts) # dynamic_hardneg_num always less than or equal use_num_hardnegs
                batch_hardneg_contexts.append(hardneg_contexts)

        if self.use_hardneg:
            # this loop is to tokenize hardneg_contexts, get hardneg_mask
            batch_hardneg_context_ids = []
            batch_hardneg_mask = []
            hardneg_padding_nums = [] # [batch_size]
            for hardneg_contexts in batch_hardneg_contexts:
                padding_len = dynamic_hardneg_num - len(hardneg_contexts)
                hardneg_padding_nums.append(padding_len)
                hardneg_mask = [1] * len(hardneg_contexts) + [0] * padding_len
                batch_hardneg_mask.append(hardneg_mask)
                per_sample_hardneg_context_ids = [] # each sample has multiple hardneg contexts, 2D list
                for hardneg_context in hardneg_contexts:
                    hardneg_context = copy.copy(hardneg_context) # make shallow copy for changing hardneg_context inplace
                    recursive_apply(hardneg_context, lambda x: x.strip()) # change hardneg_context inplace
                    hardneg_context_ids = self.tokenizer.encode(hardneg_context['title'], 
                        text_pair=hardneg_context['text'], add_special_tokens=True)
                    if len(hardneg_context_ids) > self.max_length:
                        hardneg_context_ids = hardneg_context_ids[:self.max_length]
                        hardneg_context_ids[-1] = self.tokenizer.sep_token_id
                    if max_batch_context_len < len(hardneg_context_ids):
                        max_batch_context_len = len(hardneg_context_ids)
                    per_sample_hardneg_context_ids.append(hardneg_context_ids)
                batch_hardneg_context_ids.append(per_sample_hardneg_context_ids) # 3D list
            batch_hardneg_mask = torch.tensor(batch_hardneg_mask) # [batch_size, dynamic_hardneg_num]
        
        # this loop is to padding question_ids
        batch_question_input_ids = []
        batch_question_attn_mask = []
        for question_ids in batch_question_ids:
            q_attn_mask = [1] * len(question_ids)
            padding_len = max_batch_question_len - len(question_ids)
            if padding_len > 0:
                question_ids += [self.tokenizer.pad_token_id] * padding_len
                q_attn_mask += [0] * padding_len
            batch_question_input_ids.append(question_ids)
            batch_question_attn_mask.append(q_attn_mask)
        batch_question_input_ids = torch.tensor(batch_question_input_ids) # [batch_size, max_batch_question_len]
        batch_question_attn_mask = torch.tensor(batch_question_attn_mask) # [batch_size, max_batch_question_len]
        
        # this loop is to padding positive context ids
        batch_positive_input_ids = []
        batch_positive_attn_mask = []
        for positive_context_ids in batch_positive_context_ids:
            padding_len = max_batch_context_len - len(positive_context_ids)
            c_attn_mask = [1] * len(positive_context_ids)
            if padding_len > 0:
                positive_context_ids += [self.tokenizer.pad_token_id] * padding_len
                c_attn_mask += [0] * padding_len
            batch_positive_input_ids.append(positive_context_ids)
            batch_positive_attn_mask.append(c_attn_mask)
        batch_positive_input_ids = torch.tensor(batch_positive_input_ids) # [batch_size, max_batch_context_len]
        batch_positive_attn_mask = torch.tensor(batch_positive_attn_mask) # [batch_size, max_batch_context_len]

        if self.use_hardneg:
            # this loop is to padding hardneg context ids
            batch_hardneg_input_ids = []
            batch_hardneg_attn_mask = []
            for idx, per_sample_hardneg_context_ids in enumerate(batch_hardneg_context_ids):
                per_sample_hardneg_input_ids = []
                per_sample_hardneg_attn_mask = []
                for hardneg_context_ids in per_sample_hardneg_context_ids:
                    c_attn_mask = [1] * len(hardneg_context_ids)
                    padding_len = max_batch_context_len - len(hardneg_context_ids)
                    if padding_len > 0:
                        hardneg_context_ids += [self.tokenizer.pad_token_id] * padding_len
                        c_attn_mask += [0] * padding_len
                    per_sample_hardneg_input_ids.append(hardneg_context_ids)
                    per_sample_hardneg_attn_mask.append(c_attn_mask)
                per_sample_hardneg_input_ids = torch.tensor(per_sample_hardneg_input_ids)
                per_sample_hardneg_attn_mask = torch.tensor(per_sample_hardneg_attn_mask)
                num_hardneg_pad = hardneg_padding_nums[idx]
                if num_hardneg_pad > 0:
                    per_sample_hardneg_input_ids = torch.cat(
                        [per_sample_hardneg_input_ids, 
                        torch.zeros(num_hardneg_pad, max_batch_context_len, dtype=torch.long)],
                        dim=0
                    )
                    per_sample_hardneg_attn_mask = torch.cat(
                        [per_sample_hardneg_attn_mask,
                        torch.zeros(num_hardneg_pad, max_batch_context_len, dtype=torch.long)]
                    )
                batch_hardneg_input_ids.append(per_sample_hardneg_input_ids)
                batch_hardneg_attn_mask.append(per_sample_hardneg_attn_mask)
            batch_hardneg_input_ids = torch.stack(batch_hardneg_input_ids, dim=0)
            batch_hardneg_attn_mask = torch.stack(batch_hardneg_attn_mask, dim=0)

        ids = torch.tensor([item['sample_id'] for item in items])

        batch_size = len(items)
        ids_replicate_row = ids.unsqueeze(1).repeat(1, batch_size)
        ids_replicate_col = ids.unsqueeze(0).repeat(batch_size, 1)
        duplicate_mask = ~(ids_replicate_row == ids_replicate_col) | torch.eye(batch_size, dtype=torch.bool)
        
        if self.use_hardneg:
            hardneg_mask_replicated = batch_hardneg_mask.view(1, -1).to(torch.bool).repeat(batch_size, 1)
            mask = torch.cat([duplicate_mask, hardneg_mask_replicated], dim=1)
        else:
            mask = duplicate_mask

        if self.use_hardneg:
            combine_context_input_ids = torch.cat([
                batch_positive_input_ids, batch_hardneg_input_ids.view(-1, max_batch_context_len)], dim=0)
            combine_context_attn_mask = torch.cat([batch_positive_attn_mask, 
                batch_hardneg_attn_mask.view(-1, max_batch_context_len)], dim=0)
        else:
            combine_context_input_ids = batch_positive_input_ids
            combine_context_attn_mask = batch_positive_attn_mask

        batch = {
            'question/input_ids': batch_question_input_ids,
            'question/attn_mask': batch_question_attn_mask,
            'context/input_ids': combine_context_input_ids,
            'context/attn_mask': combine_context_attn_mask,
            'mask': mask,
            'ids': ids
        }
        return batch


class WrapperGCDPRInbatchIterator(object):
    def __init__(
        self,
        num_train_epochs: int,
        train_data_iterator,
    ):
        self.num_train_epochs = num_train_epochs
        self.train_data_iterator = train_data_iterator
    
    def __iter__(self):
        for epoch in range(self.num_train_epochs):
            self.train_data_iterator.set_epoch(epoch)
            data_loader = DataLoader(self.train_data_iterator, num_workers=1, batch_size=None, shuffle=False)
            for biencoder_batch in data_loader:
                # TODO: convert to appropriate format
                question_input_ids = biencoder_batch.question_ids
                question_attn_mask = (question_input_ids > 0).to(torch.long)
                context_input_ids = biencoder_batch.context_ids
                positive_context_input_ids = context_input_ids[0::2, :]
                hardneg_context_input_ids = context_input_ids[1::2, :]
                context_input_ids = torch.cat([positive_context_input_ids, hardneg_context_input_ids], dim=0)
                context_attn_mask = (context_input_ids > 0).to(torch.long)
                mask = torch.ones(question_input_ids.size(0), context_input_ids.size(0), dtype=torch.bool)
                batch = {
                    'question/input_ids': question_input_ids,
                    'question/attn_mask': question_attn_mask,
                    'context/input_ids': context_input_ids,
                    'context/attn_mask': context_attn_mask,
                    'mask': mask
                }
                yield batch

import os
import time
import torch
import logging
import tensorflow as tf

from typing import Dict, Text
from torch.cuda.amp import GradScaler, autocast

from srdpr.constants import (
    INBATCH_PIPELINE_NAME,
    POS_PIPELINE_NAME,
    POSHARD_PIPELINE_NAME,
    HARD_PIPELINE_NAME,
    LOG_SEPARATOR
)
from srdpr.nn.losses import LossCalculator
from srdpr.nn.grad_cache import RandContext
from dpr.models.biencoder import BiEncoder
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import get_model_obj, CheckpointState
from dpr.options import get_encoder_params_state

logger = logging.getLogger(__name__)


class SRBiEncoderTrainer(object):
    def __init__(
        self,
        cfg,
        biencoder: BiEncoder,
        optimizer,
        scheduler,
        iterators,
        trained_steps: int,
    ):
        self.cfg = cfg
        self.biencoder = biencoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iterators = iterators
        for pipeline, iterator in iterators.items():
            iterator.start_worker()
        self.trained_steps = trained_steps
        self.loss_calculator = LossCalculator()
        self.scaler = GradScaler() if cfg.fp16 else None
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup data pipelines for fetching data item."""

        regulate_factor = self.cfg.regulate_factor
        possible_pipelines = [
            INBATCH_PIPELINE_NAME, POS_PIPELINE_NAME, POSHARD_PIPELINE_NAME, HARD_PIPELINE_NAME]
        available_pipelines = [
            p for p in possible_pipelines if p in self.iterators]
        both_inbatch_and_pos = all([p in available_pipelines for p in [
                                   INBATCH_PIPELINE_NAME, POS_PIPELINE_NAME]])
        if both_inbatch_and_pos:
            logger.warn("You shouldn't include both '{}' and '{}' pipelines!".format(
                INBATCH_PIPELINE_NAME, POS_PIPELINE_NAME))
            inbatch_num_continuous_steps = regulate_factor // 2
        else:
            inbatch_num_continuous_steps = (
                regulate_factor
                if INBATCH_PIPELINE_NAME in available_pipelines else 0
            )
        pos_num_continuous_steps = regulate_factor - inbatch_num_continuous_steps
        poshard_and_hard_num_continuous_steps = [
            int(p in available_pipelines) for p in [POSHARD_PIPELINE_NAME, HARD_PIPELINE_NAME]]
        num_continuous_steps = [inbatch_num_continuous_steps,
                                pos_num_continuous_steps] + poshard_and_hard_num_continuous_steps
        truth_array = [p in available_pipelines for p in possible_pipelines]
        num_continuous_steps = [num_continuous_steps[i] for i in range(
            len(num_continuous_steps)) if truth_array[i]]
        self._available_pipelines = available_pipelines
        self._index_boundaries = [sum(num_continuous_steps[:idx])
                                  for idx in range(len(num_continuous_steps) + 1)]
        self._cycle_walk = sum(num_continuous_steps)
    
    def _save_checkpoint(self, step, rng_states):
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        if not tf.io.gfile.exists(cfg.output_dir):
            tf.io.gfile.makedirs(cfg.output_dir)
        cp = os.path.join(cfg.output_dir,
                          cfg.checkpoint_file_name + '.' + str(step + 1))
        all_cp_files = tf.io.gfile.listdir(cfg.output_dir)
        all_cp_files = [os.path.join(cfg.output_dir, f) for f in all_cp_files]
        all_cp_files = sorted(all_cp_files, key=lambda x: tf.io.gfile.stat(x).mtime_nsec, reverse=True)
        if cfg.keep_checkpoint_max > 0:
            files_to_delete = all_cp_files[cfg.keep_checkpoint_max - 1:]
            for f in files_to_delete:
                tf.io.gfile.remove(f)

        meta_params = get_encoder_params_state(cfg)

        pipeline_dict = {k: v.get_idxs_generator_state()
            for k, v in self.iterators.items()}
        state = {
            'model_dict': model_to_save.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
            'pipeline_dict': pipeline_dict,
            'encoder_params': meta_params,
            'step': step + 1,
            **rng_states
        }
        with tf.io.gfile.GFile(cp, "wb") as writer:
            torch.save(state, writer)
        logger.info('Saved checkpoint at %s', cp)
        return cp
    
    def validate_and_save(self, step):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        # if epoch == args.val_av_rank_start_epoch:
        #     self.best_validation_result = None

        # if epoch >= args.val_av_rank_start_epoch:
        #     validation_loss = self.validate_average_rank()
        # else:
        #     validation_loss = self.validate_nll()

        # < save RNG state
        cpu_state = torch.get_rng_state()
        if cfg.device.type == 'cuda':
            cuda_state = torch.cuda.get_rng_state()
        else:
            cuda_state = None
        if cfg.distributed_world_size > 1:
            cuda_states = all_gather_list(cuda_state)
        else:
            cuda_states = [cuda_state]
        rng_states = {'cpu_state': cpu_state, 'cuda_states': cuda_states}
        # save RNG state />

        if save_cp:
            cp_name = self._save_checkpoint(step, rng_states=rng_states)
            logger.info('Saved checkpoint to %s', cp_name)

            # if validation_loss < (self.best_validation_result or validation_loss + 1):
            #     self.best_validation_result = validation_loss
            #     self.best_cp_name = cp_name
            #     logger.info('New Best validation checkpoint %s', cp_name)

    def run_train(self):
        if self.trained_steps > 0:
            logger.info("[Rank {}] Model has been updated for {} steps.".format(self.cfg.local_rank, self.trained_steps))
        self.biencoder.train()
        self._fetch_data_time = 0.0
        self._forward_backward_time = 0.0
        for step in range(self.trained_steps, self.cfg.total_updates):
            per_step_loss = self.train_step(step)
            if (step + 1) % self.cfg.log_batch_step == 0:
                log_string = f"[Rank {self.cfg.local_rank}] "
                step_log_string = "Step: {}/{}".format(step + 1, self.cfg.total_updates)
                log_string += step_log_string + " " * max(20 - len(step_log_string), 0) + LOG_SEPARATOR
                loss_log_string = "Loss = {}".format(per_step_loss)
                log_string += loss_log_string + " " * max(30 - len(loss_log_string), 0) + LOG_SEPARATOR
                fetch_time_log_string = "Fetch data time = {}".format(self._fetch_data_time)
                log_string += fetch_time_log_string + " " * max(40 - len(fetch_time_log_string), 0) + LOG_SEPARATOR
                fwd_bwd_time_log_string = "Forward backward time = {}".format(self._forward_backward_time)
                log_string += fwd_bwd_time_log_string
                logger.info(log_string)
            self._fetch_data_time = 0.0
            self._forward_backward_time = 0.0
            if (step + 1) % self.cfg.save_checkpoint_freq == 0:
                self.validate_and_save(step)
                self.biencoder.train()
    
    def train_step(self, step):
        pipeline = self.get_pipeline_for_step(step)
        t0 = time.perf_counter()
        batches = self.fetch_batches(pipeline)
        self._fetch_data_time += (time.perf_counter() - t0)
        computation_fn = self.get_computation_fn(pipeline)

        t0 = time.perf_counter()
        per_update_loss = 0.0
        for batch in batches:
            loss = computation_fn(batch, getattr(self.cfg.pipeline, pipeline).gradient_accumulate_steps)
            per_update_loss += loss

        per_update_loss /= len(batches)

        if self.cfg.fp16:
            self.scaler.unscale_(self.optimizer)
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), self.cfg.max_grad_norm)
        if self.cfg.fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.biencoder.zero_grad()
        self._forward_backward_time += (time.perf_counter() - t0)
        return per_update_loss

    def fetch_batches(self, pipeline):
        iterator = self.iterators[pipeline]
        batches = []
        num_batches = getattr(self.cfg.pipeline, pipeline).gradient_accumulate_steps
        for _ in range(num_batches):
            batches.append(next(iterator))
        local_batches = batches
        distributed_factor = self.cfg.distributed_world_size or 1
        if distributed_factor > 1:
            local_batches = [self._get_local_batch(batch, distributed_factor) for batch in batches]
        return local_batches
    
    def _get_local_batch(self, batch, distributed_factor):
        chunked_batch = {}
        for k, v in batch.items():
            if k in {'mask', 'duplicate_mask', 'hardneg_mask'}:
                chunked_batch[k] = v
            else:
                chunked_batch[k] = self._get_local_chunks(v, distributed_factor)[self.cfg.local_rank]
        return chunked_batch
    
    @staticmethod
    def _get_local_chunks(v, distributed_factor):
        batch_size = v.size(0)
        chunk_size = batch_size // distributed_factor
        remainder = batch_size - chunk_size * distributed_factor
        added = [1] * remainder + [0] * (distributed_factor - remainder)
        chunk_sizes = [chunk_size + added[i] for i in range(distributed_factor)]
        idx = 0
        chunks = []
        for chunk_size in chunk_sizes:
            chunks.append(v[idx: idx + chunk_size])
            idx += chunk_size
        return chunks

    def get_pipeline_for_step(self, step):
        index = step % self._cycle_walk
        for i in range(len(self._index_boundaries)):
            if self._index_boundaries[i] <= index < self._index_boundaries[i + 1]:
                break
        return self._available_pipelines[i]

    def get_computation_fn(self, pipeline):
        if pipeline == 'pos' or pipeline == 'hard':
            return self.pos_randneg_computation
        elif pipeline == 'poshard':
            return self.pos_hardneg_computation
        elif pipeline == 'inbatch':
            return self.inbatch_computation
    
    def pos_randneg_computation(self, batch: Dict[Text, torch.Tensor], gradient_accumulate_steps: int):
        question_input_ids = batch['question/input_ids'].to(self.cfg.device)
        question_attn_mask = batch['question/attn_mask'].to(self.cfg.device)
        positive_context_input_ids = batch['positive_context/input_ids'].to(self.cfg.device)
        positive_context_attn_mask = batch['positive_context/attn_mask'].to(self.cfg.device)
        negative_context_input_ids = batch['negative_context/input_ids'].to(self.cfg.device)
        negative_context_attn_mask = batch['negative_context/attn_mask'].to(self.cfg.device)
        duplicate_mask = batch['duplicate_mask'].to(self.cfg.device)

        if self.cfg.grad_cache:
            return self._pos_randneg_computation_grad_cache(question_input_ids, question_attn_mask,
                positive_context_input_ids, positive_context_attn_mask, negative_context_input_ids,
                negative_context_attn_mask, duplicate_mask, gradient_accumulate_steps)
        else:
            return self._pos_randneg_computation_no_cache(question_input_ids, question_attn_mask,
                positive_context_input_ids, positive_context_attn_mask, negative_context_input_ids,
                negative_context_attn_mask, duplicate_mask, gradient_accumulate_steps)
    
    def _pos_randneg_computation_no_cache(
        self,
        question_input_ids,
        question_attn_mask,
        positive_context_input_ids,
        positive_context_attn_mask,
        negative_context_input_ids,
        negative_context_attn_mask,
        duplicate_mask,
        gradient_accumulate_steps: int
    ):
        local_q_vector = self._get_q_representations(question_input_ids, question_attn_mask)
        local_positive_ctxs_vector = self._get_ctx_representations(positive_context_input_ids, positive_context_attn_mask)
        local_negative_ctxs_vector = self._get_ctx_representations(negative_context_input_ids, negative_context_attn_mask)
        
        # all gather
        distributed_world_size = self.cfg.distributed_world_size or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
            positive_ctx_vector_to_send = torch.empty_like(local_positive_ctxs_vector).cpu().copy_(local_positive_ctxs_vector).detach_()
            negative_ctx_vector_to_send = torch.empty_like(local_negative_ctxs_vector).cpu().copy_(local_negative_ctxs_vector).detach_()

            global_question_ctx_vectors = all_gather_list(
                [q_vector_to_send, positive_ctx_vector_to_send, negative_ctx_vector_to_send],
                max_size=self.cfg.global_loss_buf_sz)

            global_q_vector = []
            global_positive_ctxs_vector = []
            global_negative_ctxs_vector = []

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, positive_ctxs_vector, negative_ctxs_vector = item

                if i != self.cfg.local_rank:
                    global_q_vector.append(q_vector.to(local_q_vector.device))
                    global_positive_ctxs_vector.append(positive_ctxs_vector.to(local_q_vector.device))
                    global_negative_ctxs_vector.append(negative_ctxs_vector.to(local_q_vector.device))
                else:
                    global_q_vector.append(local_q_vector)
                    global_positive_ctxs_vector.append(local_positive_ctxs_vector)
                    global_negative_ctxs_vector.append(local_negative_ctxs_vector)

            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_positive_ctxs_vector = torch.cat(global_positive_ctxs_vector, dim=0)
            global_negative_ctxs_vector = torch.cat(global_negative_ctxs_vector, dim=0)
        else:
            global_q_vector = local_q_vector
            global_positive_ctxs_vector = local_positive_ctxs_vector
            global_negative_ctxs_vector = local_negative_ctxs_vector

        loss = self.loss_calculator.compute(
            inputs=
            {
                'question_embeddings': global_q_vector,
                'positive_context_embeddings': global_positive_ctxs_vector,
                'negative_context_embeddings': global_negative_ctxs_vector,
                'duplicate_mask': duplicate_mask
            },
            sim_func=self.cfg.sim_func,
            compute_type='pos_randneg')

        distributed_factor = self.cfg.distributed_world_size or 1
        _loss = loss * (distributed_factor / self.cfg.loss_scale)
        if gradient_accumulate_steps > 1:
            _loss = _loss / gradient_accumulate_steps
        if self.cfg.fp16:
            self.scaler.scale(_loss).backward()
        else:
            _loss.backward()

        return loss.item()

    def _pos_randneg_computation_grad_cache(
        self,
        question_input_ids,
        question_attn_mask,
        positive_context_input_ids,
        positive_context_attn_mask,
        negative_context_input_ids,
        negative_context_attn_mask,
        duplicate_mask,
        gradient_accumulate_steps: int
    ):
        forward_batch_size = positive_context_input_ids.size(0)
        contrastive_size = negative_context_input_ids.size(0)

        combine_context_input_ids = torch.cat([
            positive_context_input_ids, negative_context_input_ids], dim=0)
        combine_context_attn_mask = torch.cat([
            positive_context_attn_mask, negative_context_attn_mask], dim=0)
        
        # 1. forward no grad
        q_id_chunks = question_input_ids.split(self.cfg.q_chunk_size)
        q_attn_mask_chunks = question_attn_mask.split(self.cfg.q_chunk_size)

        ctx_id_chunks = combine_context_input_ids.split(self.cfg.ctx_chunk_size)
        ctx_attn_mask_chunks = combine_context_attn_mask.split(self.cfg.ctx_chunk_size)

        all_q_reps = []
        all_ctx_reps = []

        q_rnds = []
        c_rnds = []

        for id_chunk, attn_chunk in zip(q_id_chunks, q_attn_mask_chunks):
            q_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                if self.cfg.fp16:
                    with autocast():
                        q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                else:
                    q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
            all_q_reps.append(q_chunk_reps)
        all_q_reps = torch.cat(all_q_reps)

        for id_chunk, attn_chunk in zip(
                ctx_id_chunks, ctx_attn_mask_chunks):
            c_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                if self.cfg.fp16:
                    with autocast():
                        ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                else:
                    ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
            all_ctx_reps.append(ctx_chunk_reps)
        all_ctx_reps = torch.cat(all_ctx_reps)
        
        # 2. loss calculation
        all_q_reps = all_q_reps.float().detach().requires_grad_()
        all_ctx_reps = all_ctx_reps.float().detach().requires_grad_()
        all_positive_ctx_reps = all_ctx_reps[:forward_batch_size]
        all_negative_ctx_reps = all_ctx_reps[forward_batch_size:]

        # all gather
        distributed_world_size = self.cfg.distributed_world_size or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(all_q_reps).cpu().copy_(all_q_reps).detach_()
            positive_ctx_vector_to_send = torch.empty_like(all_positive_ctx_reps).cpu().copy_(all_positive_ctx_reps).detach_()
            negative_ctx_vector_to_send = torch.empty_like(all_negative_ctx_reps).cpu().copy_(all_negative_ctx_reps).detach_()

            global_question_ctx_vectors = all_gather_list(
                [q_vector_to_send, positive_ctx_vector_to_send, negative_ctx_vector_to_send],
                max_size=self.cfg.global_loss_buf_sz)
            
            global_q_vector = []
            global_positive_ctxs_vector = []
            global_negative_ctxs_vector = []

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, positive_ctxs_vector, negative_ctxs_vector = item

                if i != self.cfg.local_rank:
                    global_q_vector.append(q_vector.to(all_q_reps.device))
                    global_positive_ctxs_vector.append(positive_ctxs_vector.to(all_q_reps.device))
                    global_negative_ctxs_vector.append(negative_ctxs_vector.to(all_q_reps.device))
                else:
                    global_q_vector.append(all_q_reps)
                    global_positive_ctxs_vector.append(all_positive_ctx_reps)
                    global_negative_ctxs_vector.append(all_negative_ctx_reps)

            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_positive_ctxs_vector = torch.cat(global_positive_ctxs_vector, dim=0)
            global_negative_ctxs_vector = torch.cat(global_negative_ctxs_vector, dim=0)
        else:
            global_q_vector = all_q_reps
            global_positive_ctxs_vector = all_positive_ctx_reps
            global_negative_ctxs_vector = all_negative_ctx_reps
        
        loss = self.loss_calculator.compute(
            inputs=
            {
                'question_embeddings': global_q_vector,
                'positive_context_embeddings': global_positive_ctxs_vector,
                'negative_context_embeddings': global_negative_ctxs_vector,
                'duplicate_mask': duplicate_mask
            },
            sim_func=self.cfg.sim_func,
            compute_type='pos_randneg')

        distributed_factor = self.cfg.distributed_world_size or 1
        _loss = loss * (distributed_factor / self.cfg.loss_scale)
        if gradient_accumulate_steps > 1:
            _loss = _loss / gradient_accumulate_steps
        if self.cfg.fp16:
            self.scaler.scale(_loss).backward()
        else:
            _loss.backward()
        
        # 3. chunk forward backward
        q_grads = all_q_reps.grad.split(self.cfg.q_chunk_size)
        ctx_grads = all_ctx_reps.grad.split(self.cfg.ctx_chunk_size)

        for id_chunk, attn_chunk, grad, rnd in zip(
                q_id_chunks, q_attn_mask_chunks, q_grads, q_rnds):
            with rnd:
                if self.cfg.fp16:
                    with autocast():
                        q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                        surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())
                else:
                    q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                    surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())

            if self.cfg.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        for id_chunk, attn_chunk, grad, rnd in zip(
                ctx_id_chunks, ctx_attn_mask_chunks, ctx_grads, c_rnds):
            with rnd:
                if self.cfg.fp16:
                    with autocast():
                        ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                        surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())
                else:
                    ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                    surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())

            if self.cfg.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        return loss.item()

    def pos_hardneg_computation(self, batch: Dict[Text, torch.Tensor], gradient_accumulate_steps):
        question_input_ids = batch['question/input_ids'].to(self.cfg.device)
        question_attn_mask = batch['question/attn_mask'].to(self.cfg.device)
        positive_context_input_ids = batch['positive_context/input_ids'].to(self.cfg.device)
        positive_context_attn_mask = batch['positive_context/attn_mask'].to(self.cfg.device)
        hardneg_context_input_ids = batch['hardneg_context/input_ids'].to(self.cfg.device)
        hardneg_context_attn_mask = batch['hardneg_context/attn_mask'].to(self.cfg.device)
        hardneg_mask = batch['hardneg_mask'].to(self.cfg.device)

        if self.cfg.grad_cache:
            return self._pos_hardneg_computation_grad_cache(question_input_ids, question_attn_mask,
                positive_context_input_ids, positive_context_attn_mask, hardneg_context_input_ids,
                hardneg_context_attn_mask, hardneg_mask, gradient_accumulate_steps)
        else:
            return self._pos_hardneg_computation_no_cache(question_input_ids, question_attn_mask,
                positive_context_input_ids, positive_context_attn_mask, hardneg_context_input_ids,
                hardneg_context_attn_mask, hardneg_mask, gradient_accumulate_steps)

    def _pos_hardneg_computation_no_cache(
        self,
        question_input_ids,
        question_attn_mask,
        positive_context_input_ids,
        positive_context_attn_mask,
        hardneg_context_input_ids,
        hardneg_context_attn_mask,
        hardneg_mask,
        gradient_accumulate_steps: int
    ):
        forward_batch_size, contrastive_size, ctx_seq_len = hardneg_context_input_ids.size()
        hardneg_context_input_ids_2d = hardneg_context_input_ids.view(-1, ctx_seq_len)
        hardneg_context_attn_mask_2d = hardneg_context_attn_mask.view(-1, ctx_seq_len)

        local_q_vector = self._get_q_representations(question_input_ids, question_attn_mask)
        local_positive_ctxs_vector = self._get_ctx_representations(positive_context_input_ids, positive_context_attn_mask)
        local_hardneg_ctxs_vector = self._get_ctx_representations(hardneg_context_input_ids_2d, hardneg_context_attn_mask_2d)
        
        # all gather
        distributed_world_size = self.cfg.distributed_world_size or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
            positive_ctx_vector_to_send = torch.empty_like(local_positive_ctxs_vector).cpu().copy_(local_positive_ctxs_vector).detach_()
            hardneg_ctx_vector_to_send = torch.empty_like(local_hardneg_ctxs_vector).cpu().copy_(local_hardneg_ctxs_vector).detach_()

            global_question_ctx_vectors = all_gather_list(
                [q_vector_to_send, positive_ctx_vector_to_send, hardneg_ctx_vector_to_send],
                max_size=self.cfg.global_loss_buf_sz)

            global_q_vector = []
            global_positive_ctxs_vector = []
            global_hardneg_ctxs_vector = []

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, positive_ctxs_vector, hardneg_ctxs_vector = item

                if i != self.cfg.local_rank:
                    global_q_vector.append(q_vector.to(local_q_vector.device))
                    global_positive_ctxs_vector.append(positive_ctxs_vector.to(local_q_vector.device))
                    global_hardneg_ctxs_vector.append(hardneg_ctxs_vector.view(forward_batch_size, contrastive_size, -1).to(local_q_vector.device))
                else:
                    global_q_vector.append(local_q_vector)
                    global_positive_ctxs_vector.append(local_positive_ctxs_vector)
                    global_hardneg_ctxs_vector.append(local_hardneg_ctxs_vector.view(forward_batch_size, contrastive_size, -1))

            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_positive_ctxs_vector = torch.cat(global_positive_ctxs_vector, dim=0)
            global_hardneg_ctxs_vector = torch.cat(global_hardneg_ctxs_vector, dim=0)
        else:
            global_q_vector = local_q_vector
            global_positive_ctxs_vector = local_positive_ctxs_vector
            global_hardneg_ctxs_vector = local_hardneg_ctxs_vector.view(forward_batch_size, contrastive_size, -1)

        loss = self.loss_calculator.compute(
            inputs={
                'question_embeddings': global_q_vector,
                'positive_context_embeddings': global_positive_ctxs_vector,
                'hardneg_context_embeddings': global_hardneg_ctxs_vector,
                'hardneg_mask': hardneg_mask
            },
            sim_func=self.cfg.sim_func,
            compute_type='pos_hardneg')

        distributed_factor = self.cfg.distributed_world_size or 1
        _loss = loss * (distributed_factor / self.cfg.loss_scale)
        if gradient_accumulate_steps > 1:
            _loss = _loss / gradient_accumulate_steps
        if self.cfg.fp16:
            self.scaler.scale(_loss).backward()
        else:
            _loss.backward()
        
        return loss.item()
    
    def _pos_hardneg_computation_grad_cache(
        self,
        question_input_ids,
        question_attn_mask,
        positive_context_input_ids,
        positive_context_attn_mask,
        hardneg_context_input_ids,
        hardneg_context_attn_mask,
        hardneg_mask,
        gradient_accumulate_steps: int
    ):
        forward_batch_size, contrastive_size, ctx_seq_len = hardneg_context_input_ids.size()
        hardneg_context_input_ids_2d = hardneg_context_input_ids.view(-1, ctx_seq_len)
        hardneg_context_attn_mask_2d = hardneg_context_attn_mask.view(-1, ctx_seq_len)

        combine_context_input_ids = torch.cat([
            positive_context_input_ids, hardneg_context_input_ids_2d], dim=0)
        combine_context_attn_mask = torch.cat([
            positive_context_attn_mask, hardneg_context_attn_mask_2d], dim=0)
        
        # 1. forward no grad
        q_id_chunks = question_input_ids.split(self.cfg.q_chunk_size)
        q_attn_mask_chunks = question_attn_mask.split(self.cfg.q_chunk_size)

        ctx_id_chunks = combine_context_input_ids.split(self.cfg.ctx_chunk_size)
        ctx_attn_mask_chunks = combine_context_attn_mask.split(self.cfg.ctx_chunk_size)

        all_q_reps = []
        all_ctx_reps = []

        q_rnds = []
        c_rnds = []

        for id_chunk, attn_chunk in zip(q_id_chunks, q_attn_mask_chunks):
            q_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                if self.cfg.fp16:
                    with autocast():
                        q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                else:
                    q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
            all_q_reps.append(q_chunk_reps)
        all_q_reps = torch.cat(all_q_reps)

        for id_chunk, attn_chunk in zip(
                ctx_id_chunks, ctx_attn_mask_chunks):
            c_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                if self.cfg.fp16:
                    with autocast():
                        ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                else:
                    ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
            all_ctx_reps.append(ctx_chunk_reps)
        all_ctx_reps = torch.cat(all_ctx_reps)

        # 2. loss calculation
        all_q_reps = all_q_reps.float().detach().requires_grad_()
        all_ctx_reps = all_ctx_reps.float().detach().requires_grad_()
        all_positive_ctx_reps = all_ctx_reps[:forward_batch_size]
        all_hardneg_ctx_reps = all_ctx_reps[forward_batch_size:]

        # all gather
        distributed_world_size = self.cfg.distributed_world_size or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(all_q_reps).cpu().copy_(all_q_reps).detach_()
            positive_ctx_vector_to_send = torch.empty_like(all_positive_ctx_reps).cpu().copy_(all_positive_ctx_reps).detach_()
            hardneg_ctx_vector_to_send = torch.empty_like(all_hardneg_ctx_reps).cpu().copy_(all_hardneg_ctx_reps).detach_()

            global_question_ctx_vectors = all_gather_list(
                [q_vector_to_send, positive_ctx_vector_to_send, hardneg_ctx_vector_to_send],
                max_size=self.cfg.global_loss_buf_sz)
            
            global_q_vector = []
            global_positive_ctxs_vector = []
            global_hardneg_ctxs_vector = []

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, positive_ctxs_vector, hardneg_ctxs_vector = item

                if i != self.cfg.local_rank:
                    global_q_vector.append(q_vector.to(all_q_reps.device))
                    global_positive_ctxs_vector.append(positive_ctxs_vector.to(all_q_reps.device))
                    global_hardneg_ctxs_vector.append(hardneg_ctxs_vector.view(forward_batch_size, contrastive_size, -1).to(all_q_reps.device))
                else:
                    global_q_vector.append(all_q_reps)
                    global_positive_ctxs_vector.append(all_positive_ctx_reps)
                    global_hardneg_ctxs_vector.append(all_hardneg_ctx_reps.view(forward_batch_size, contrastive_size, -1))

            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_positive_ctxs_vector = torch.cat(global_positive_ctxs_vector, dim=0)
            global_hardneg_ctxs_vector = torch.cat(global_hardneg_ctxs_vector, dim=0)
        else:
            global_q_vector = all_q_reps
            global_positive_ctxs_vector = all_positive_ctx_reps
            global_hardneg_ctxs_vector = all_hardneg_ctx_reps.view(forward_batch_size, contrastive_size, -1)

        loss = self.loss_calculator.compute(
            inputs=
            {
                'question_embeddings': global_q_vector,
                'positive_context_embeddings': global_positive_ctxs_vector,
                'hardneg_context_embeddings': global_hardneg_ctxs_vector,
                'hardneg_mask': hardneg_mask
            },
            sim_func=self.cfg.sim_func,
            compute_type='pos_hardneg')

        distributed_factor = self.cfg.distributed_world_size or 1
        _loss = loss * (distributed_factor / self.cfg.loss_scale)
        if gradient_accumulate_steps > 1:
            _loss = _loss / gradient_accumulate_steps
        if self.cfg.fp16:
            self.scaler.scale(_loss).backward()
        else:
            _loss.backward()
        
        # 3. chunk forward backward
        q_grads = all_q_reps.grad.split(self.cfg.q_chunk_size)
        ctx_grads = all_ctx_reps.grad.split(self.cfg.ctx_chunk_size)

        for id_chunk, attn_chunk, grad, rnd in zip(
                q_id_chunks, q_attn_mask_chunks, q_grads, q_rnds):
            with rnd:
                if self.cfg.fp16:
                    with autocast():
                        q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                        surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())
                else:
                    q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                    surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())

            if self.cfg.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        for id_chunk, attn_chunk, grad, rnd in zip(
                ctx_id_chunks, ctx_attn_mask_chunks, ctx_grads, c_rnds):
            with rnd:
                if self.cfg.fp16:
                    with autocast():
                        ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                        surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())
                else:
                    ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                    surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())

            if self.cfg.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()
        
        return loss.item()
    
    def inbatch_computation(self, batch: Dict[Text, torch.Tensor], gradient_accumulate_steps: int):
        question_input_ids = batch['question/input_ids'].to(self.cfg.device)
        question_attn_mask = batch['question/attn_mask'].to(self.cfg.device)
        context_input_ids = batch['context/input_ids'].to(self.cfg.device)
        context_attn_mask = batch['context/attn_mask'].to(self.cfg.device)
        mask = batch['mask'].to(self.cfg.device)

        if self.cfg.grad_cache:
            return self._inbatch_computation_grad_cache(question_input_ids, question_attn_mask,
                context_input_ids, context_attn_mask, mask, gradient_accumulate_steps)
        else:
            return self._inbatch_computation_no_cache(question_input_ids, question_attn_mask,
                context_input_ids, context_attn_mask, mask, gradient_accumulate_steps)        
    
    def _inbatch_computation_no_cache(
        self,
        question_input_ids,
        question_attn_mask,
        context_input_ids,
        context_attn_mask,
        mask,
        gradient_accumulate_steps
    ):
        local_q_vector = self._get_q_representations(question_input_ids, question_attn_mask)
        local_ctxs_vector = self._get_ctx_representations(context_input_ids, context_attn_mask)

        # all gather
        distributed_world_size = self.cfg.distributed_world_size or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
            ctx_vector_to_send = torch.empty_like(local_ctxs_vector).cpu().copy_(local_ctxs_vector).detach()

            global_question_ctx_vectors = all_gather_list(
                [q_vector_to_send, ctx_vector_to_send],
                max_size=self.cfg.global_loss_buf_sz)

            global_q_vector = []
            global_ctxs_vector = []

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, ctxs_vector = item

                if i != self.cfg.local_rank:
                    global_q_vector.append(q_vector.to(local_q_vector.device))
                    global_ctxs_vector.append(ctxs_vector.to(local_q_vector.device))
                else:
                    global_q_vector.append(local_q_vector)
                    global_ctxs_vector.append(local_ctxs_vector)

            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)
        else:
            global_q_vector = local_q_vector
            global_ctxs_vector = local_ctxs_vector
        
        loss = self.loss_calculator.compute(
            inputs=
            {
                'question_embeddings': global_q_vector,
                'context_embeddings': global_ctxs_vector,
                'mask': mask
            },
            sim_func=self.cfg.sim_func,
            compute_type='inbatch')
    
        distributed_factor = self.cfg.distributed_world_size or 1
        _loss = loss * (distributed_factor / self.cfg.loss_scale)
        if gradient_accumulate_steps > 1:
            _loss = _loss / gradient_accumulate_steps
        if self.cfg.fp16:
            self.scaler.scale(_loss).backward()
        else:
            _loss.backward()

        return loss.item()

    def _inbatch_computation_grad_cache(
        self,
        question_input_ids,
        question_attn_mask,
        context_input_ids,
        context_attn_mask,
        mask,
        gradient_accumulate_steps
    ):
        # 1. forward no grad
        q_id_chunks = question_input_ids.split(self.cfg.q_chunk_size)
        q_attn_mask_chunks = question_attn_mask.split(self.cfg.q_chunk_size)

        ctx_id_chunks = context_input_ids.split(self.cfg.ctx_chunk_size)
        ctx_attn_mask_chunks = context_attn_mask.split(self.cfg.ctx_chunk_size)

        all_q_reps = []
        all_ctx_reps = []

        q_rnds = []
        c_rnds = []

        for id_chunk, attn_chunk in zip(q_id_chunks, q_attn_mask_chunks):
            q_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                if self.cfg.fp16:
                    with autocast():
                        q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                else:
                    q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
            all_q_reps.append(q_chunk_reps)
        all_q_reps = torch.cat(all_q_reps)

        for id_chunk, attn_chunk in zip(
                ctx_id_chunks, ctx_attn_mask_chunks):
            c_rnds.append(RandContext(id_chunk, attn_chunk))
            with torch.no_grad():
                if self.cfg.fp16:
                    with autocast():
                        ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                else:
                    ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
            all_ctx_reps.append(ctx_chunk_reps)
        all_ctx_reps = torch.cat(all_ctx_reps)

        # 2. loss calculation
        all_q_reps = all_q_reps.float().detach().requires_grad_()
        all_ctx_reps = all_ctx_reps.float().detach().requires_grad_()

        # all gather
        distributed_world_size = self.cfg.distributed_world_size or 1
        if distributed_world_size > 1:
            q_vector_to_send = torch.empty_like(all_q_reps).cpu().copy_(all_q_reps).detach_()
            ctx_vector_to_send = torch.empty_like(all_ctx_reps).cpu().copy_(all_ctx_reps).detach_()

            global_question_ctx_vectors = all_gather_list(
                [q_vector_to_send, ctx_vector_to_send],
                max_size=self.cfg.global_loss_buf_sz)
            
            global_q_vector = []
            global_ctxs_vector = []

            for i, item in enumerate(global_question_ctx_vectors):
                q_vector, ctxs_vector = item

                if i != self.cfg.local_rank:
                    global_q_vector.append(q_vector.to(all_q_reps.device))
                    global_ctxs_vector.append(ctxs_vector.to(all_q_reps.device))
                else:
                    global_q_vector.append(all_q_reps)
                    global_ctxs_vector.append(all_ctx_reps)

            global_q_vector = torch.cat(global_q_vector, dim=0)
            global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)
        else:
            global_q_vector = all_q_reps
            global_ctxs_vector = all_ctx_reps
        
        loss = self.loss_calculator.compute(
            inputs=
            {
                'question_embeddings': global_q_vector,
                'context_embeddings': global_ctxs_vector,
                'mask': mask
            },
            sim_func=self.cfg.sim_func,
            compute_type='inbatch')

        distributed_factor = self.cfg.distributed_world_size or 1
        _loss = loss * (distributed_factor / self.cfg.loss_scale)
        if gradient_accumulate_steps > 1:
            _loss = _loss / gradient_accumulate_steps
        if self.cfg.fp16:
            self.scaler.scale(_loss).backward()
        else:
            _loss.backward()

        # 3. chunk forward backward
        q_grads = all_q_reps.grad.split(self.cfg.q_chunk_size)
        ctx_grads = all_ctx_reps.grad.split(self.cfg.ctx_chunk_size)

        for id_chunk, attn_chunk, grad, rnd in zip(
                q_id_chunks, q_attn_mask_chunks, q_grads, q_rnds):
            with rnd:
                if self.cfg.fp16:
                    with autocast():
                        q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                        surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())
                else:
                    q_chunk_reps = self._get_q_representations(id_chunk, attn_chunk)
                    surrogate = torch.dot(q_chunk_reps.flatten().float(), grad.flatten())

            if self.cfg.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()

        for id_chunk, attn_chunk, grad, rnd in zip(
                ctx_id_chunks, ctx_attn_mask_chunks, ctx_grads, c_rnds):
            with rnd:
                if self.cfg.fp16:
                    with autocast():
                        ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                        surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())
                else:
                    ctx_chunk_reps = self._get_ctx_representations(id_chunk, attn_chunk)
                    surrogate = torch.dot(ctx_chunk_reps.flatten().float(), grad.flatten())

            if self.cfg.fp16:
                self.scaler.scale(surrogate).backward()
            else:
                surrogate.backward()
        
        return loss.item()

    def _get_ctx_representations(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        if self.cfg.fp16:
            with autocast():
                ctxs_vector = self.biencoder(
                    None, None, None,
                    input_ids, None, attn_mask
                )[1]
        else:
            ctxs_vector = self.biencoder(
                None, None, None,
                input_ids, None, attn_mask
            )[1]
        return ctxs_vector
    
    def _get_q_representations(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        if self.cfg.fp16:
            with autocast():
                q_vector = self.biencoder(
                    input_ids, None, attn_mask,
                    None, None, None,
                )[0]
        else:
            q_vector = self.biencoder(
                input_ids, None, attn_mask,
                None, None, None,
            )[0]
        return q_vector

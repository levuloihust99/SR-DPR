import logging
import os
import sys
import hydra
import glob
from omegaconf import DictConfig

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder
from dpr.options import set_encoder_params_from_state, setup_args_gpu, set_seed, print_args
from dpr.utils.data_utils import ShardedDataIterableDataset, read_data_from_json_files

from dpr.utils.model_utils import get_model_file, get_model_obj, get_schedule_linear, load_states_from_checkpoint, setup_for_distributed_mode
from srdpr.constants import HARD_PIPELINE_NAME, HARD_SEED, INBATCH_PIPELINE_NAME, INBATCH_SEED, POSHARD_PIPELINE_NAME, POSHARD_SEED
from srdpr.data_helpers.dataloader import HardDataIterator, InbatchDataIterator, PoshardDataIterator, StatelessIdxsGenerator, WrapperGCDPRInbatchIterator
from srdpr.data_helpers.datasets import ByteDataset
from srdpr.utils.logging_utils import add_color_formatter
from srdpr.utils.helpers import dictconfig_to_namespace
from srdpr.nn.trainer import SRBiEncoderTrainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    add_color_formatter(logging.root)
    global logger
    logger = logging.getLogger(__name__)

    cfg = dictconfig_to_namespace(cfg)

    setup_args_gpu(cfg)
    set_seed(cfg)
    print_args(cfg)

    logger.info("***** Initializing components for training *****")
    # if model file is specified, encoder parameters from saved state should be used for initialization
    model_file = get_model_file(cfg, cfg.checkpoint_file_name)
    saved_state = None
    if model_file:
        saved_state = load_states_from_checkpoint(model_file)
        set_encoder_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, biencoder, optimizer = init_biencoder_components(cfg.encoder_model_type, cfg)

    biencoder, optimizer = setup_for_distributed_mode(biencoder, optimizer, cfg.device, cfg.n_gpu,
                                                    cfg.local_rank,
                                                    cfg.fp16,
                                                    cfg.fp16_opt_level)
    
    dataset = ByteDataset(data_path=cfg.bytedataset, idx_record_size=6)
    pipelines_to_build = cfg.train_mode.split('+')
    iterators = {}
    if POSHARD_PIPELINE_NAME in pipelines_to_build:
        iterators[POSHARD_PIPELINE_NAME] = PoshardDataIterator(
            dataset=dataset,
            idxs_generator=StatelessIdxsGenerator(len(dataset), shuffle_seed=cfg.seed + POSHARD_SEED),
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, POSHARD_PIPELINE_NAME).forward_batch_size,
            contrastive_size=getattr(cfg.pipeline, POSHARD_PIPELINE_NAME).contrastive_size,
            max_length=cfg.max_length
        )
    if HARD_PIPELINE_NAME in pipelines_to_build:
        iterators[HARD_PIPELINE_NAME] = HardDataIterator(
            hardneg_dataset=dataset,
            hardneg_idxs_generator=StatelessIdxsGenerator(len(dataset), shuffle_seed=cfg.seed + HARD_SEED),
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, HARD_PIPELINE_NAME).forward_batch_size,
            contrastive_size=getattr(cfg.pipeline, HARD_PIPELINE_NAME).contrastive_size,
            max_length=cfg.max_length
        )
    if INBATCH_PIPELINE_NAME in pipelines_to_build:
        iterators[INBATCH_PIPELINE_NAME] = InbatchDataIterator(
            dataset=dataset,
            idxs_generator=StatelessIdxsGenerator(len(dataset), shuffle_seed=cfg.seed + INBATCH_SEED),
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, INBATCH_PIPELINE_NAME).forward_batch_size,
            use_hardneg=getattr(cfg.pipeline, INBATCH_PIPELINE_NAME).use_hardneg,
            use_num_hardnegs=getattr(cfg.pipeline, INBATCH_PIPELINE_NAME).use_num_hardnegs,
            max_length=cfg.max_length,
        )

    scheduler = get_schedule_linear(optimizer, cfg.warmup_steps, cfg.total_updates)
    trained_steps = 0
    if saved_state is not None:
        # Load saved state
        model_to_load = get_model_obj(biencoder)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection

        if saved_state.optimizer_dict:
            logger.info('Loading saved optimizer state ...')
            optimizer.load_state_dict(saved_state.optimizer_dict)
        
        if saved_state.scheduler_dict:
            scheduler_state = saved_state.scheduler_dict
            logger.info("Loading scheduler state %s", scheduler_state)
            shift = int(scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                optimizer,
                cfg.warmup_steps,
                cfg.total_updates,
                steps_shift=shift
            )
        
        if saved_state.pipeline_dict: 
            pipeline_state = saved_state.pipeline_dict
            for pipeline in pipelines_to_build:
                specific_pipeline_state = pipeline_state.get(pipeline)
                if specific_pipeline_state is None:
                    logger.warning("You are continuing training with a different pipeline setup.")
                else:
                    epoch = specific_pipeline_state['epoch']
                    iteration = specific_pipeline_state['iteration']
                    if pipeline in {'pos', 'hard'}:
                        iteration = iteration - specific_pipeline_state['shift_back']
                    iterators[pipeline].set_idxs_generator_state(epoch, iteration)
        
        if saved_state.step:
            trained_steps = saved_state.step

    summary_writer = SummaryWriter(cfg.log_dir)
    trainer = SRBiEncoderTrainer(
        cfg=cfg,
        biencoder=biencoder,
        optimizer=optimizer,
        scheduler=scheduler,
        iterators=iterators,
        summary_writer=summary_writer,
        trained_steps=trained_steps
    )

    trainer.run_train()


if __name__ == "__main__":
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    main()

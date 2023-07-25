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
from srdpr.constants import (
    HARD_PIPELINE_NAME, HARD_SEED,
    INBATCH_PIPELINE_NAME, INBATCH_SEED,
    POSHARD_PIPELINE_NAME, POSHARD_SEED,
    POS_PIPELINE_NAME, POS_SEED
)
from srdpr.data_helpers.dataloader import (
    PosDataIterator,
    PoshardDataIterator,
    HardDataIterator,
    InbatchDataIterator,
    StatelessIdxsGenerator
)
from srdpr.data_helpers.datasets import ByteDataset
from srdpr.utils.logging_utils import add_color_formatter
from srdpr.utils.helpers import dictconfig_to_namespace
from srdpr.nn.trainer import SRBiEncoderTrainer


def encode_decorator(encode):
    """This decorator is to add behaviour when title of a context is None."""

    def wrapper(
        text,
        text_pair = None,
        add_special_tokens = True,
        padding = False,
        truncation = False,
        max_length = None,
        stride: int = 0,
        return_tensors = None,
        **kwargs
    ):
        if text is None:
            text = text_pair
            text_pair = None
        return encode(
            text, text_pair, add_special_tokens,
            padding, truncation, max_length,
            stride, return_tensors, **kwargs
        )
    return wrapper


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
        encoder_params = getattr(saved_state, 'encoder_params', None)
        if encoder_params:
            set_encoder_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, biencoder, optimizer = init_biencoder_components(cfg.encoder_model_type, cfg, add_pooling_layer=False)
    tensorizer.tokenizer.encode = encode_decorator(tensorizer.tokenizer.encode)

    biencoder, optimizer = setup_for_distributed_mode(biencoder, optimizer, cfg.device, cfg.n_gpu,
                                                    cfg.local_rank,
                                                    cfg.fp16,
                                                    cfg.fp16_opt_level)

    pipelines_to_build = cfg.train_mode.split('+')
    iterators = {}
    if POS_PIPELINE_NAME in pipelines_to_build:
        pos_dataset = ByteDataset(data_path=cfg.bytedataset.pos, idx_record_size=6)
        iterators[POS_PIPELINE_NAME] = PosDataIterator(
            dataset=pos_dataset,
            idxs_generator=StatelessIdxsGenerator(len(pos_dataset), shuffle_seed=cfg.seed + POS_SEED),
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, POS_PIPELINE_NAME).forward_batch_size,
            contrastive_size=getattr(cfg.pipeline, POS_PIPELINE_NAME).contrastive_size,
            max_length=cfg.max_length
        )
    if POSHARD_PIPELINE_NAME in pipelines_to_build:
        poshard_dataset = ByteDataset(data_path=cfg.bytedataset.poshard, idx_record_size=6)
        iterators[POSHARD_PIPELINE_NAME] = PoshardDataIterator(
            dataset=poshard_dataset,
            idxs_generator=StatelessIdxsGenerator(len(poshard_dataset), shuffle_seed=cfg.seed + POSHARD_SEED),
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, POSHARD_PIPELINE_NAME).forward_batch_size,
            contrastive_size=getattr(cfg.pipeline, POSHARD_PIPELINE_NAME).contrastive_size,
            max_length=cfg.max_length
        )
    if HARD_PIPELINE_NAME in pipelines_to_build:
        hardneg_dataset = ByteDataset(data_path=cfg.bytedataset.hard.hardneg, idx_record_size=6)
        if getattr(cfg.pipeline, HARD_PIPELINE_NAME).use_randneg:
            randneg_dataset = ByteDataset(data_path=cfg.bytedataset.hard.randneg, idx_record_size=6)
            randneg_idxs_generator = StatelessIdxsGenerator(len(randneg_dataset), shuffle_seed=cfg.seed + HARD_SEED)
        else:
            randneg_dataset = None
            randneg_idxs_generator = None
        iterators[HARD_PIPELINE_NAME] = HardDataIterator(
            hardneg_dataset=hardneg_dataset,
            hardneg_idxs_generator=StatelessIdxsGenerator(len(hardneg_dataset), shuffle_seed=cfg.seed + HARD_SEED),
            randneg_dataset=randneg_dataset,
            randneg_idxs_generator=randneg_idxs_generator,
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, HARD_PIPELINE_NAME).forward_batch_size,
            contrastive_size=getattr(cfg.pipeline, HARD_PIPELINE_NAME).contrastive_size,
            max_length=cfg.max_length,
            use_randneg_dataset=getattr(cfg.pipeline, HARD_PIPELINE_NAME).use_randneg
        )
    if INBATCH_PIPELINE_NAME in pipelines_to_build:
        inbatch_dataset = ByteDataset(data_path=cfg.bytedataset.inbatch, idx_record_size=6)
        iterators[INBATCH_PIPELINE_NAME] = InbatchDataIterator(
            dataset=inbatch_dataset,
            idxs_generator=StatelessIdxsGenerator(len(inbatch_dataset), shuffle_seed=cfg.seed + INBATCH_SEED),
            tokenizer=tensorizer.tokenizer,
            forward_batch_size=getattr(cfg.pipeline, INBATCH_PIPELINE_NAME).forward_batch_size,
            use_hardneg=getattr(cfg.pipeline, INBATCH_PIPELINE_NAME).use_hardneg,
            use_num_hardnegs=getattr(cfg.pipeline, INBATCH_PIPELINE_NAME).use_num_hardnegs,
            max_length=cfg.max_length,
        )

    scheduler = get_schedule_linear(optimizer, cfg.warmup_steps, cfg.total_updates)
    trained_steps = 0
    history = None
    if saved_state is not None:
        # Load saved state
        model_to_load = get_model_obj(biencoder)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection

        optimizer_state = getattr(saved_state, 'optimizer_dict', None)
        if optimizer_state:
            logger.info('Loading saved optimizer state ...')
            optimizer.load_state_dict(optimizer_state)

        scheduler_state = getattr(saved_state, 'scheduler_dict', None)
        if scheduler_state:
            logger.info("Loading scheduler state %s", scheduler_state)
            scheduler.load_state_dict(scheduler_state)
            # shift = int(scheduler_state["last_epoch"])
            # logger.info("Steps shift %d", shift)
            # scheduler = get_schedule_linear(
            #     optimizer,
            #     cfg.warmup_steps,
            #     cfg.total_updates,
            #     steps_shift=shift
            # )

        pipeline_state = getattr(saved_state, 'pipeline_dict', None)
        if pipeline_state:
            for pipeline in pipelines_to_build:
                specific_pipeline_state = pipeline_state.get(pipeline)
                if specific_pipeline_state is None:
                    logger.warning("You are continuing training with a different pipeline setup.")
                else:
                    iterators[pipeline].set_idxs_generator_state(specific_pipeline_state)
        
        # restore RNG
        cpu_state = getattr(saved_state, 'cpu_state', None)
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        
        cuda_states = getattr(saved_state, 'cuda_states', None)
        if cuda_states is not None:
            if torch.cuda.is_available() > 0:
                device_id = torch.cuda.current_device()
                if device_id < len(cuda_states):
                    try:
                        torch.cuda.set_rng_state(cuda_states[device_id])
                    except Exception:
                        logger.info("Invalid RNG state restored from checkpoint file '{}'".format(model_file))

        saved_state_step = getattr(saved_state, 'step', None)
        if saved_state_step:
            trained_steps = saved_state_step

        saved_state_history = getattr(saved_state, 'history', None)
        if saved_state_history:
            history = saved_state_history

    summary_writer = SummaryWriter(cfg.log_dir)
    trainer = SRBiEncoderTrainer(
        cfg=cfg,
        biencoder=biencoder,
        optimizer=optimizer,
        scheduler=scheduler,
        iterators=iterators,
        summary_writer=summary_writer,
        trained_steps=trained_steps,
        history=history,
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

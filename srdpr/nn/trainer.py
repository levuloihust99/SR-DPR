import logging
from srdpr.constants import (
    INBATCH_PIPELINE_NAME,
    POS_PIPELINE_NAME,
    POSHARD_PIPELINE_NAME,
    HARD_PIPELINE_NAME
)

logger = logging.getLogger(__name__)


class SRBiEncoderTrainer(object):
    def __init__(self, cfg, iterators):
        self.cfg = cfg
        self.iterators = iterators
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
        self._cycle_walk = regulate_factor + \
            sum(poshard_and_hard_num_continuous_steps)

    def run_train(self):
        for step in range(self.cfg.num_updates):
            self.train_step(step)
    
    def train_step(self, step):
        pipeline = self.get_pipeline_for_step(step)
        batches = self.fetch_batches(pipeline)
    
    def fetch_batches(self, pipeline):
        iterator = self.iterators[pipeline]
        batches = []
        num_batches = getattr(self.cfg, pipeline).gradient_accumulate_steps
        for _ in range(num_batches):
            batches.append(next(iterator))
        return batches

    def get_pipeline_for_step(self, step):
        index = step % self._cycle_walk
        for i in range(len(self._index_boundaries)):
            if self._index_boundaries[i] <= index < self._index_boundaries[i + 1]:
                break
        return self._available_pipelines[i]

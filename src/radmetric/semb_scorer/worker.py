import numpy as np
import ray

from .model import SEMBScorer
from ..reward_server import RewardServer


@ray.remote(num_gpus=1)
class SEMBScoreWorker:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.scorer = SEMBScorer(batch_size=batch_size).cuda(device='cuda:0')

    def compute(self, hyps, refs):
        f1 = self.scorer.score(hyps=hyps, refs=refs)
        return np.array(f1)


class SEMBScoreMetric(RewardServer):
    def __init__(
        self,
        num_workers=None,
        batch_size=16,
    ):
        self.batch_size = batch_size

        super().__init__(num_workers=num_workers)

    def create_worker(self):
        return SEMBScoreWorker.remote(batch_size=self.batch_size)

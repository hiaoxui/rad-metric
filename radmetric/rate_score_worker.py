import numpy as np
import torch
import ray

from .reward_server import RewardServer


@ray.remote(num_gpus=1)
class RaTEScoreWorker:
    def __init__(self, batch_size=8):
        from RaTEScore import RaTEScore
        self.model = RaTEScore(batch_size=batch_size, use_gpu=True)

    def compute(self, hyps, refs):
        with torch.inference_mode():
            reward_list = self.model.compute_score(candidate_list=hyps, reference_list=refs)
        return np.array(reward_list)


class RaTEScoreMetric(RewardServer):
    def __init__(self, num_workers=None, batch_size=8):
        self.batch_size = batch_size

        super().__init__(num_workers=num_workers)

    def create_worker(self):
        return RaTEScoreWorker.remote(batch_size=self.batch_size)

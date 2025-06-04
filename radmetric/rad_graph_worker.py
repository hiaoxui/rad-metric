import ray
from radgraph import F1RadGraph

from .reward_server import RewardServer


@ray.remote(num_gpus=1)
class RadGraphWorker:
    def __init__(self, reward_level="all", batch_size=8):
        self.model = F1RadGraph(
            reward_level=reward_level,
            cuda=0,
            batch_size=batch_size
        )

    def compute(self, hyps, refs):
        reward_list = self.model(hyps=hyps, refs=refs)[1]
        rewards = list(zip(*reward_list))
        return list(rewards)


class RadGraphRewardServer(RewardServer):
    def __init__(self, num_workers=None, reward_level="all", batch_size=8):
        ray.init(ignore_reinit_error=True)
        self.batch_size = batch_size
        self.reward_level = reward_level

        super().__init__(num_workers=num_workers)

    def create_worker(self):
        return RadGraphWorker.remote(
            reward_level=self.reward_level,
            batch_size=self.batch_size
        )

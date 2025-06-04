from typing import Dict, Any, List, Optional

import ray

from .f1chexbert import F1CheXbert
from .reward_server import RewardServer


@ray.remote(num_gpus=1)
class F1CheXbertWorker:
    def __init__(self):
        self.model = F1CheXbert()

    def compute(self, hyps: List[str], refs: List[str]) -> List[Dict[str, Any]]:
        return self.model(hyps=hyps, refs=refs)

class F1CheXbertRewardServer(RewardServer):
    def __init__(self, num_workers: Optional[int] = None, batch_size: int = 8):
        ray.init(ignore_reinit_error=True)
        self.batch_size = batch_size
        super().__init__(num_workers=num_workers)

    def create_worker(self):
        return F1CheXbertWorker.remote()

    def compute_rewards(self, hyps: List[str], refs: List[str]):
        assert len(hyps) == len(refs), "hypotheses and references must be same length"
        total = len(hyps)
        batch_size = self.batch_size
        futures = []

        for i in range(self.num_workers):
            start = i * batch_size
            end = min((i + 1) * batch_size, total)
            if start < end:
                futures.append(self.workers[i].compute.remote(hyps[start:end], refs[start:end]))

        results = ray.get(futures)
        # each result is a list of refs_chexbert_5, hyps_chexbert_5, ref_labels, hyp_labels
        refs_chexbert_5 = [batch[0] for batch in results]
        hyps_chexbert_5 = [batch[1] for batch in results]
        ref_labels = [batch[2] for batch in results]
        hyp_labels = [batch[3] for batch in results]

        return F1CheXbert.report_results(
            refs_chexbert_5=refs_chexbert_5,
            hyps_chexbert_5=hyps_chexbert_5,
            ref_labels=ref_labels,
            hyp_labels=hyp_labels
        )

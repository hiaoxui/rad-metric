from typing import Dict, Any, List, Optional, Tuple

import ray

from .f1chexbert import F1CheXbert
from ..reward_server import RewardServer


@ray.remote(num_gpus=1)
class F1CheXbertWorker:
    def __init__(self):
        self.model = F1CheXbert()

    def compute(self, hyps: List[str], refs: List[str]) -> Tuple[List[int], List[int]]:
        return self.model.batch_run(hyps=hyps, refs=refs)


class F1CheXbertMetric(RewardServer):
    def __init__(self, num_workers: Optional[int] = None, batch_size: int = 8):
        self.batch_size = batch_size
        super().__init__(num_workers=num_workers)

    def create_worker(self):
        return F1CheXbertWorker.remote()

    def run_in_batch(self, hyps: List[str], refs: List[str]) -> Tuple[List[int], List[int]]:
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
        refs12, hyps12 = [], []
        for res in results:
            refs12.extend(res[0])
            hyps12.extend(res[1])
        return refs12, hyps12

    def compute_rewards(self, hyps: List[str], refs: List[str]):
        refs12, hyps12 = self.run_in_batch(hyps, refs)
        # report f1/acc score and accuracy for individual predictions
        return F1CheXbert.report_individual(refs12, hyps12)

    def compute_metrics(self, hyps: List[str], refs: List[str]) -> Dict[str, Any]:
        refs12, hyps12 = self.run_in_batch(hyps, refs)
        accuracy, pe_accuracy, cr, cr_5 = F1CheXbert.report_results(refs12, hyps12)
        return {
            "accuracy": accuracy,
            "pe_accuracy": pe_accuracy,
            "cr": cr,
            "cr_5": cr_5
        }

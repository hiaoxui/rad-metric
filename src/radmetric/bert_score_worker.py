import ray

from .reward_server import RewardServer


@ray.remote(num_gpus=1)
class BERTScoreWorker:
    def __init__(
        self,
        lang="en",
        batch_size=16,
        model_type="distilroberta-base",
    ):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(
            model_type=model_type,
            batch_size=batch_size,
            lang=lang,
            rescale_with_baseline=True,
            idf=False,
            device="cuda:0",
        )

    def compute(self, hyps, refs):
        _, _, f1 = self.scorer.score(cands=hyps, refs=refs)
        return f1.numpy()


class BERTScoreMetric(RewardServer):
    def __init__(
        self,
        num_workers=None,
        batch_size=16,
        model_type="distilroberta-base",
        lang="en"
    ):
        self.batch_size = batch_size
        self.model_type = model_type
        self.lang = lang

        super().__init__(num_workers=num_workers)

    def create_worker(self):
        return BERTScoreWorker.remote(
            batch_size=self.batch_size,
            model_type=self.model_type,
            lang=self.lang,
        )

import os


class BLEUMetric:
    def compute_metrics(self, hyps, refs):
        import evaluate
        bleu = evaluate.load('bleu')
        return bleu.compute(predictions=hyps, references=refs)['bleu']

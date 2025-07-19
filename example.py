from typing import List

import numpy as np

from radmetric import (
    RadGraphMetric,
    SEMBScoreMetric,
    BERTScoreMetric,
    F1CheXbertMetric,
    RaTEScoreMetric,
    BLEUMetric,
)


def test_rad_graph_metric(refs: List[str], hyps: List[str]):
    metric = RadGraphMetric()
    results = metric.compute_rewards(hyps, refs)
    # above is a list of f1 scores
    average_f1 = np.mean(results)
    print(f"RadGraphMetric results: {average_f1}")


def test_bert_score_metric(refs: List[str], hyps: List[str]):
    metric = BERTScoreMetric()
    results = metric.compute_rewards(hyps, refs)
    avg = np.mean(np.array(results))
    print(f"BERTScoreMetric results: {avg:.4f}")


def test_semb_score_metric(refs: List[str], hyps: List[str]):
    metric = SEMBScoreMetric()
    results = metric.compute_rewards(hyps, refs)
    avg = np.mean(np.array(results))
    print(f"SEMBScoreMetric results: {avg:.4f}")


def test_f1_chexbert_metric(refs: List[str], hyps: List[str]):
    metric = F1CheXbertMetric()
    results = metric.compute_metrics(hyps, refs)
    macro_avg_f1 = results['cr']['macro avg']['f1-score']
    print(f"F1CheXbertMetric results: {macro_avg_f1:.4f}")


def test_rate_score_metric(refs: List[str], hyps: List[str]):
    metric = RaTEScoreMetric()
    results = metric.compute_rewards(hyps, refs)
    avg = np.mean(np.array(results))
    print(f"RaTEScoreMetric results: {avg:.4f}")


def test_bleu_metric(refs: List[str], hyps: List[str]):
    metric = BLEUMetric()
    results = metric.compute_metrics(hyps, refs)
    avg = np.mean(np.array(results))
    print(f"BLEUMetric results: {avg:.4f}")


def test():
    # prepare 32 pairs of samples
    references = [
        'Findings: As compared to the previous radiograph, there is no relevant change.\nExtensive right pleural effusion, potentially combined with some degree of pleural thickening, relatively extensive atelectatic changes in the right lung bases.\nThe extent of the ventilated lung parenchyma on the right is small and located around the right perihilar areas.\nUnremarkable left heart border, moderate tortuosity of the thoracic aorta.\nNormal appearance of the left lung without evidence of parenchymal changes or left pleural effusion. Impression: None',
        'Findings: PA and lateral views of the chest are compared to previous exam from ___.\nCompared to prior, there has been no significant interval change.\nThere is no evidence of focal consolidation.\nIncreased interstitial markings on one of the lateral views resolves on the second lateral view, likely due to improved inspiratory effort.\nCardiomediastinal silhouette is unchanged, as are the osseous and soft tissue structures.\nCalcific densities projecting over the neck and left upper quadrant are unchanged, as are the vascular stents. Impression: No evidence of acute cardiopulmonary process.',
    ] * 16
    hypotheses = [
        'Findings:As compared to the previous radiograph, there is no relevant change.  The large right pleural effusion with subsequent atelectasis is constant in appearance.  The left lung is unchanged.  Normal size of the cardiac silhouette.  No left pleural effusion.  No pneumothorax.  The right PICC line has been removed in the interval.Impression:No relevant change in appearance of the large right pleural effusion.',
        'Findings:AP and lateral views of the chest.  The lungs are clear of focal consolidation or effusion.  The cardiomediastinal silhouette is within normal limits.  No acute osseous abnormality is identified.Impression:No acute cardiopulmonary process.'
    ] * 16

    test_bleu_metric(refs=references, hyps=hypotheses)
    test_rad_graph_metric(refs=references, hyps=hypotheses)
    test_bert_score_metric(refs=references, hyps=hypotheses)
    test_semb_score_metric(refs=references, hyps=hypotheses)
    test_f1_chexbert_metric(refs=references, hyps=hypotheses)
    test_rate_score_metric(refs=references, hyps=hypotheses)



if __name__ == "__main__":
    test()
    print("All tests passed.")

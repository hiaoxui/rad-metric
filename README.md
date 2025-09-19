# rad-metric

Composite metrics for chest X-ray report (CXR) generation.

Supported metrics:
- BLEU
- BertScore
- SembScore
- CheXbert
- RadGraph
- RaTEScore

## Setup

```bash
pip install rad-metric
```

## Usage

We use `ray` to initialize the evaluation workers, CPU for BLEU and GPU for the rest.
By default, it will use all the available GPU devices in the current machine.
If you want to run this metric software on multi-node cluster, initialize ray yourself and
add more nodes as you need.

Refer to [example.py](/example.py) for example usages.


## Reference

Please prioritize to cite the original contributors of each metric.

BLEU:
```bibtex
@inproceedings{bleu02,
  year = {2002},
  url = {https://doi.org/10.3115/1073083.1073135},
  author = {Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)},
  title = {{{BLEU}}: A Method for Automatic Evaluation of Machine Translation}
}
```

RadGraph
```bibtex
@inproceedings{jainRadGraphExtractingClinical2021,
  year = {2021},
  url = {https://doi.org/10.48550/arXiv.2106.14463},
  author = {Jain, Saahil and Agrawal, Ashwin and Saporta, Adriel and Truong, Steven QH and Duong, Du Nguyen and Bui, Tan and Chambon, Pierre and Zhang, Yuhao and Lungren, Matthew P. and Ng, Andrew Y. and Langlotz, Curtis P. and Rajpurkar, Pranav},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  title = {{{RadGraph}}: {{Extracting Clinical Entities}} and {{Relations}} from {{Radiology Reports}}}
}
```

SembScore and F1CheXbert
```bibtex
@inproceedings{smitCheXbertCombiningAutomatic2020,
  year = {2020},
  url = {https://doi.org/10.48550/arXiv.2004.09167},
  author = {Smit, Akshay and Jain, Saahil and Rajpurkar, Pranav and Pareek, Anuj and Ng, Andrew Y. and Lungren, Matthew P.},
  booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  title = {{{CheXbert}}: {{Combining Automatic Labelers}} and {{Expert Annotations}} for {{Accurate Radiology Report Labeling Using BERT}}}
}
```

RaTEScore
```bibtex
@inproceedings{zhaoRaTEScoreMetricRadiology2024,
  year = {2024},
  url = {https://doi.org/10.18653/v1/2024.emnlp-main.836},
  author = {Zhao, Weike and Wu, Chaoyi and Zhang, Xiaoman and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  title = {{{RaTEScore}}: {{A Metric}} for {{Radiology Report Generation}}}
}
```

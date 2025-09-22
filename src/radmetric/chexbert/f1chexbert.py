# This file is modified from f1chexbert
from typing import List, Tuple
import re
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics._classification import _check_targets

REPO_ID = 'StanfordAIMI/RRG_scorers'
FILE_NAME = 'chexbert.pth'

TARGET_NAMES_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding",
]
TARGET_NAMES_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
TARGET_NAMES_5_INDEX = np.where(np.isin(TARGET_NAMES_14, TARGET_NAMES_5))[0]

class BertLabeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None, inference=False, **kwargs):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(BertLabeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif inference:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        # size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape
                                            (batch_size, 4) and the last has shape (batch_size, 2)
        """
        # shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


class F1CheXbert(nn.Module):
    def __init__(self, device='cuda:0'):
        super(F1CheXbert, self).__init__()
        self.device = torch.device(device)
        # Model and tok
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertLabeler(inference=True)

        # Downloading pretrain model from huggingface
        file_path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME)

        # Load model
        state_dict = torch.load(file_path, map_location=self.device)['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v

        # Load params
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    @staticmethod
    def _clean_text(text: str):
        if not isinstance(text, str):
            return ''
        else:
            return re.sub(r'\s+', ' ', text).strip()

    @torch.inference_mode()
    def get_label(self, texts: List[str], mode="rrg"):
        bsz = len(texts)
        texts = list(map(self._clean_text, texts))
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        ).to(self.device)
        model_out = self.model(inputs['input_ids'], inputs['attention_mask'])
        out_indices = [out.to(device='cpu').argmax(dim=1).tolist() for out in model_out]

        ret = []
        for i_batch in range(bsz):
            batch_out = [oi[i_batch] for oi in out_indices]
            ret.append(self._extract_label(batch_out, mode))
        return ret

    @staticmethod
    def _extract_label(out, mode):
        v = []
        if mode == "rrg":
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
            v = [int(isinstance(l, int) and l > 0) for l in v]

        elif mode == "classification":
            # https://github.com/stanfordmlgroup/CheXbert/blob/master/src/label.py#L124
            for c in out:
                if c == 0:
                    v.append('')
                if c == 3:
                    v.append(-1)
                if c == 2:
                    v.append(0)
                if c == 1:
                    v.append(1)
        else:
            raise NotImplementedError(mode)

        return v

    @torch.inference_mode()
    def batch_run(self, hyps: List[str], refs: List[str], mode="rrg") -> Tuple[List[int], List[int]]:
        assert len(hyps) == len(refs), "hypotheses and references must be same length"
        texts = hyps + refs
        labels = self.get_label(texts, mode=mode)
        hyps12 = labels[:len(hyps)]
        refs12 = labels[len(hyps):]
        # return all 12 classes
        return refs12, hyps12

    @staticmethod
    def report_individual(refs12: List[int], hyps12: List[int]):
        # report f1/acc score and accuracy for individual predictions
        assert len(refs12) == len(hyps12), "hypotheses and references must be same length"
        metrics = []
        for ref12, hyp12 in zip(refs12, hyps12):
            accuracy, pe_accuracy, cr, cr_5 = F1CheXbert.report_results([ref12], [hyp12])
            metrics.append([accuracy, pe_accuracy, cr, cr_5])
        return metrics

    @staticmethod
    def report_results(refs12: List[int], hyps12: List[int]):
        # report f1 score and accuracy over all predictions
        assert len(refs12) == len(hyps12), "hypotheses and references must be same length"
        refs5 = [np.array(r)[TARGET_NAMES_5_INDEX] for r in refs12]
        hyps5 = [np.array(h)[TARGET_NAMES_5_INDEX] for h in hyps12]
        accuracy = accuracy_score(y_true=refs5, y_pred=hyps5)
        y_type, y_true, y_pred = _check_targets(refs5, hyps5)
        # Accuracy
        # Per element accuracy
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        pe_accuracy = (differing_labels == 0).astype(np.float32)

        cr = classification_report(
            y_true=refs12,
            y_pred=hyps12,
            target_names=TARGET_NAMES_14,
            output_dict=True,
            zero_division=1.0,
        )
        cr_5 = classification_report(
            y_true=refs5,
            y_pred=hyps5,
            target_names=TARGET_NAMES_5,
            output_dict=True,
            zero_division=1.0,
        )

        return accuracy, pe_accuracy, cr, cr_5

import argparse
from collections import OrderedDict

from tqdm import tqdm
import torch

from chexbert import utils
from chexbert.models.bert_labeler import bert_labeler
from chexbert.datasets.unlabeled_dataset import UnlabeledDataset
from chexbert.constants import *


def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    idx_list = [s['idx'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list, 'idx': idx_list}
    return batch

def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not

    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader

def label(checkpoint_path, csv_path, filename="data.pt", logits=False): # TODO: CHECK WITH VISH ABOUT filename
    """Labels a dataset of reports
    @param checkpoint_path (string): location of saved model checkpoint
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report
    """
    ld = load_unlabeled_data(csv_path)

    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    if torch.cuda.device_count() > 0: #works even if only 1 GPU available
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model) #to utilize multiple GPU's
        model = model.to(device)

    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]
    rep = {}

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            if logits:
                for idx, j in zip(data['idx'], range(len(data['idx']))):
                    rep[idx] = [torch.softmax(out[k][j], dim=0)[0].item() for k in range(len(out))]
            else:
                for idx, j in zip(data['idx'], range(len(out))):
                    rep[idx] = out[j].to('cpu')
                    #curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                    #y_pred[j].append(curr_y_pred)

        if i % 1000 == 0:
            torch.save(rep, filename) #torch.save(rep, 'data.pt') # TODO: CHECK WITH VISH
        #for j in range(len(y_pred)):
        #    y_pred[j] = torch.cat(y_pred[j], dim=0)

    #if was_training:
    #    model.train()
    torch.save(rep, filename) #torch.save(rep, 'data.pt') # TODO: CHECK WITH VISH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=False, default="data.pt", # TODO: CHECK WITH VISH
                        help='path to intended output file')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    parser.add_argument('-l', '--logits', type=bool, nargs='?', required=False, default=False, help="get logits")
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    do_logits = args.logits
    checkpoint_path = args.checkpoint

    label(checkpoint_path, csv_path, out_path, do_logits)
    #save_preds(y_pred, csv_path, out_path)

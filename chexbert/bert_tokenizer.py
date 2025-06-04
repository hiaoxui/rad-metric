import pandas as pd
import json
from tqdm import tqdm

def get_impressions_from_csv(path):
    df = pd.read_csv(path)
    imp = df['report']
    imp = imp.str.strip()
    imp = imp.replace(r'\s+', ' ', regex=True)
    imp = imp.str.strip()
    return imp

def tokenize(impressions, tokenizer):
    new_impressions = []
    print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
    for i in tqdm(range(impressions.shape[0])):
        impr = impressions.iloc[i]
        if pd.isna(impr):
            impr = ''
        tokenized_imp = tokenizer.tokenize(impr)
        if tokenized_imp: # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512: # length exceeds maximum size
                # print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else: # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions

def load_list(path):
    with open(path, 'r') as filehandle:
        impressions = json.load(filehandle)
        return impressions
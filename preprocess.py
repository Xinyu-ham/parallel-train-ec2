# Description: This file contains the code to preprocess the data and create the dataset for training the model.

import torch
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from transformers import AutoTokenizer

import pandas as pd

class TextDataset(IterDataPipe):
    def __init__(self, tokenizer: AutoTokenizer, s3_data_prefix: str, num_files: int) -> None:
        self.tokenizer = tokenizer
        self.s3_data_prefix = s3_data_prefix
        self.num_files = num_files

        self.url_wrapper = IterableWrapper([self.s3_data_prefix]).list_files_by_s3().shuffle().sharding_filter()

    def __iter__(self):
        for _, file in self.url_wrapper.load_files_by_s3():
            temp = pd.read_csv(file)
            label = torch.from_numpy(temp['outcome'].values)

            bert_input = []
            tokens = [
                self.tokenizer(t, padding='max_length', max_length=100, truncation=True, return_tensors='pt')
                for t in temp['headlines']
            ]
            bert_input.append(torch.cat([e['input_ids'] for e in tokens], dim=0))
            bert_input.append(torch.cat([e['attention_mask'] for e in tokens], dim=0))

            tabular_input = [
                torch.from_numpy(temp[col].values).to(torch.float32).squeeze() 
                for col in temp.columns 
                if col not in ['outcome', 'headlines']
            ]
            yield bert_input, tabular_input, label

    
from transformers import LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pds
import torch
import argparse
from tqdm import trange

def get_args_parser():
    parser = argparse.ArgumentParser('Pretraining', add_help=False)
    parser.add_argument('--tokenizer_path', type=str, default="tokenizer.model")
    parser.add_argument('--size', type=str, default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument('--train_mode', type=str, default="mixed", choices=["mixed", "pure"])
    parser.add_argument('--use_augment', type=bool, default=False)
    parser.add_argument('--qa_ratio', type=float, default=4.)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

class BIOPreTrain(Dataset):
    def __init__(self, args):
        self.root_path = f"data/{args.size}_augmented" if args.use_augment else f"data/{args.size}_balanced"
        self.use_qa = True if args.train_mode == 'mixed' else False
        self.qa_ratio = args.qa_ratio
        self.epochs = args.epochs
        self.data_size_one_epoch = 0

        self.data = torch.tensor([])
        for epoch in range(self.epochs):
            bio_data_path = f"{self.root_path}/bio_tokens_{epoch}.pt"
            bio_data = torch.load(bio_data_path, weights_only=True)
            epoch_data = bio_data
            if self.use_qa:
                qa_data_path = f"{self.root_path}/qa_tokens_{epoch}.pt"
                qa_data = torch.load(qa_data_path, weights_only=True)
                expected_qa_num = int(bio_data.size(0) * self.qa_ratio)
                epoch_data = torch.cat((bio_data, qa_data[:expected_qa_num,:]), dim=0)
            epoch_data = epoch_data[torch.randperm(epoch_data.size(0))]
            self.data_size_one_epoch = epoch_data.size(0)
            self.data = torch.cat((self.data, epoch_data), dim=0)
        self.data = self.data.long()

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_data = self.data[idx]
        return input_data


class BIOVal(Dataset):
    def __init__(self, args):
        self.root_path = f"data/{args.size}_augmented" if args.use_augment else f"data/{args.size}_balanced"
        self.use_qa = True
        
        self.max_words = 512
        self.tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, pad_token='[PAD]')
        
        
        bio_data_path = f"{self.root_path}/test_bios.parquet"
        bio_data = pds.read_parquet(bio_data_path)['content'].tolist()
        if self.use_qa:
            qa_data_path = self.root_path + '/test_qas.parquet'
            qa_data = pds.read_parquet(qa_data_path)['content'].tolist()
        else:
            qa_data = []

        self.data = self.tokenizer(
            bio_data + qa_data, 
            max_length=self.max_words, 
            pad_to_max_length=True, 
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data['input_ids'][idx]
        attention_mask = self.data['attention_mask'][idx]
        return input_ids, attention_mask
    
if __name__ == '__main__':
    args = get_args_parser()
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, pad_token='[PAD]')
    dataset = BIOVal(args)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    for batch in dataloader:
        input_ids, attention_mask = batch
        print(attention_mask)
        print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        break
    
    # dataset = BIOVal(args)
    # tokenizer.batch_decode(dataset[0][0])
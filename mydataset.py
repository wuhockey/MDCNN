from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os
from audio import transform_to_nlp,transform_to_cv
from transformers import BertTokenizer, BertModel
class My_Dataset(data.Dataset):
    def __init__(self,raw_data_path):
        print('Reading data:')
        self.tokenizer =  BertTokenizer.from_pretrained('./bert-base-uncased')
        self.raw_data_path= raw_data_path
        self.features,self.labels = self.get_features_labels()


    def get_features_labels(self):
        features = []
        labels = []
        dir_list = os.listdir(self.raw_data_path)
        from tqdm import tqdm
        for class_num,class_dir in enumerate(tqdm(dir_list)):
            class_dir_path = os.path.join(self.raw_data_path,class_dir)
            for _,_,file_name in os.walk(class_dir_path):
                for file in file_name:
                    file_path = os.path.join(class_dir_path ,file)
                    labels.append(class_num)
                    features.append([transform_to_cv(file_path),torch.tensor(self.tokenizer(transform_to_nlp(file_path),max_length=256,padding='max_length')['input_ids'],dtype=torch.long)])

        return features, labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item][0],self.features[item][1],self.labels[item]

if __name__ == '__main__':
    dataset = My_Dataset('datasets/trainset')
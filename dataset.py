import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, model_tokenizer):
        self.data = pd.read_excel(file_path)
        self.texts = self.data['text']
        self.labels = self.data['label']
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.model_tokenizer = model_tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        inputs = self.model_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        return {'input_ids': inputs['input_ids'][0], 'attention_mask': inputs['attention_mask'][0], 'label': torch.tensor(label)}
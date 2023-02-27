import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from utils import *


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences = []
        with open(data_path, 'r') as f:
            for line in f:
                label, sentence = line.strip().split('\t')
                self.labels.append(int(label))
                self.sentences.append(sentence)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.sentences[index],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'token_type_ids': inputs['token_type_ids'][0],
            'label': torch.tensor(self.labels[index], dtype=torch.long)
        }


class TextClassifier:
    def __init__(self, model_path, checkpoint_path, data_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=2
        )
        self.dataset = TextDataset(data_path, self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=16)

    def train(self, epochs=3, loss_func='ce', save_path=None, load_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(self.dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps, eta_min=1e-6
        )
        if loss_func == 'ce':
            criterion = torch.nn.CrossEntropyLoss()
        elif loss_func == 'focal':
            criterion = FocalLoss()
        elif loss_func == 'dist_balanced':
            criterion = DistributionBalancedLoss()
        else:
            raise ValueError('Invalid loss function name')

        if load_path:
            state_dict = torch.load(load_path)
            self.model.load_state_dict(state_dict)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = criterion(outputs.logits, batch['label'])
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')

            if save_path:
                torch.save(self.model.state_dict(), save_path)

    def predict(self, sentence):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs.to(device))
            probs = F.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs)
        return pred_label.item()


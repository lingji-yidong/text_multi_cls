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

    
import pandas as pd
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)  # Load data from xlsx file
        self.labels = self.data['labels'].unique()  # Get unique label values
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}  # Map labels to integers
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.data.loc[index, 'labels']
        text = self.data.loc[index, 'text']
        label_id = self.label_to_id[label]  # Convert label to integer
        return {'text': text, 'label': label_id}
    
    
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenizer(text)
        return tokenized_text, label

# Define the texts and labels
texts = ["this is a sample text", "this is another sample text", "yet another sample text", "one more sample text"]
labels = [0, 1, 0, 1]

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a tokenizer function
def tokenizer(text):
    return text.split()

# Create instances of the TextDataset class for the training and validation sets
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

# Define dataloaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

